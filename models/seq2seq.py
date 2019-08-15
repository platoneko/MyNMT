from modules.rnn import GRUEncoder
from modules.rnn import StackGRUDecoder
from modules.attention import MLPAttention
from modules.utils import *
from models.base_model import BaseModel
from utils.pack import Pack
from utils.metrics import accuracy, perplexity
from torch.nn.utils import clip_grad_norm_
from modules.criterions import SequenceCrossEntropy
import torch.nn as nn
from overrides import overrides


class Seq2Seq(BaseModel):
    def __init__(
            self,
            post_embedding,
            response_embedding,
            embedding_size,
            hidden_size,
            start_index,
            end_index,
            padding_index,
            num_layers=2,
            dropout=0.2,
            teaching_force_rate=0.5,
            mmi_anti=False,
            anti_gamma=1,
            anti_rate=0.5
    ):
        super().__init__()
        self.post_embedding = post_embedding
        self.response_embedding = response_embedding
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.start_index = start_index
        self.end_index = end_index
        self.padding_index = padding_index
        self.dropout = dropout
        self.teaching_force_rate = teaching_force_rate
        self.num_layers = num_layers

        self.encoder = GRUEncoder(embedding_size, hidden_size//2,
                                  dropout=dropout, num_layers=num_layers, bidirectional=True)

        decoder_attn = MLPAttention(hidden_size, hidden_size, hidden_size)
        decoder_input_size = embedding_size + hidden_size
        vocab_size = response_embedding.weight.size(0)
        self.decoder = StackGRUDecoder(
            decoder_input_size,
            hidden_size,
            vocab_size,
            start_index,
            end_index,
            response_embedding,
            num_layers=num_layers,
            attention=decoder_attn,
            dropout=dropout
        )
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=padding_index, reduction='mean')
        self.mmi_anti = mmi_anti
        self.anti_gamma = anti_gamma
        self.anti_rate = anti_rate

    def forward(self, inputs, is_training=True):
        """
        train and eval
        """
        if is_training:
            assert inputs.response is not None
        if hasattr(inputs, 'response'):
            if isinstance(inputs.response, tuple):
                response_tokens, response_len = inputs.response
            else:
                response_tokens = inputs.response
        post_tokens, post_len = inputs.post
        embedded_post = self.post_embedding(post_tokens)
        encoder_output, encoder_hidden = self.encoder((embedded_post, post_len))
        encoder_output_mask = post_tokens.ne(self.padding_index)
        if is_training:
            logits = self.decoder(
                encoder_hidden,
                target=response_tokens,
                attn_value=encoder_output,
                attn_mask=encoder_output_mask,
                teaching_force_rate=self.teaching_force_rate,
                is_training=is_training
            )
        else:
            # test
            logits = self.decoder(
                encoder_hidden,
                target=None,
                attn_value=encoder_output,
                attn_mask=encoder_output_mask,
                is_training=False
            )
        outputs = Pack(logits=logits)
        return outputs

    def beam_search(self, inputs, beam_size=4, per_node_beam_size=4):
        # designed for test or interactive mode
        post_tokens, post_len = inputs.post
        embedded_post = self.post_embedding(post_tokens)
        encoder_output, encoder_hidden = self.encoder((embedded_post, post_len))
        encoder_output_mask = post_tokens.ne(self.padding_index)
        all_top_k_predictions, log_probabilities = \
            self.decoder.forward_beam_search(encoder_hidden,
                                             attn_value=encoder_output,
                                             attn_mask=encoder_output_mask,
                                             beam_size=beam_size,
                                             per_node_beam_size=per_node_beam_size)
        prediction = all_top_k_predictions[:, 0, :]
        outputs = Pack(prediction=prediction)
        return outputs

    @overrides
    def collect_metrics(self, outputs, target):
        """
        collect_metrics
        """
        num_samples = target.size(0)
        metrics = Pack(num_samples=num_samples)
        loss = 0
        logits = outputs.logits
        nll = self.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        loss += nll
        # num_tokens = target.ne(self.padding_index).sum().item()
        prediction = logits.argmax(dim=2)
        acc = accuracy(prediction, target, padding_idx=self.padding_index, end_idx=self.end_index)
        metrics.add(nll=nll, acc=acc)
        metrics.add(ppx=math.exp(nll.item()))
        '''
        if self.mmi_anti:
            gamma_logits = logits[:, :self.anti_gamma, :]
            gamma_target = target[:, :self.anti_gamma]
            loss -= self.anti_rate * \
                    self.cross_entropy(gamma_logits.reshape(-1, logits.size(-1)),
                                       gamma_target.reshape(-1))
        loss = loss / num_tokens
        '''
        metrics.add(loss=loss)
        return metrics

    @overrides
    def iterate(self, inputs, optimizer=None, grad_clip=None):
        """
        iterate
        """
        outputs = self.forward(inputs, is_training=True)
        response_tokens, response_len = inputs.response
        target = response_tokens[:, 1:]
        metrics = self.collect_metrics(outputs, target)

        loss = metrics.loss
        if torch.isnan(loss):
            raise ValueError("nan loss encountered")

        if self.training:
            assert optimizer is not None
            optimizer.zero_grad()
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                clip_grad_norm_(parameters=self.parameters(),
                                max_norm=grad_clip)
            optimizer.step()
        return metrics
