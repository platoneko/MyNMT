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
        num_classes = response_embedding.weight.size(0)
        self.decoder = StackGRUDecoder(
            decoder_input_size,
            hidden_size,
            num_classes,
            start_index,
            end_index,
            response_embedding,
            num_layers=num_layers,
            attention=decoder_attn,
            dropout=dropout
        )
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=padding_index)

    def forward(self, inputs, is_training=True):
        """
        train and eval
        """
        if is_training:
            assert inputs.response is not None
        if hasattr(inputs, 'response'):
            response_token, response_len = inputs.response

        post_token, post_len = inputs.post
        embedded_post = self.post_embedding(post_token)
        encoder_outputs, encoder_hidden = self.encoder((embedded_post, post_len))
        encoder_outputs_mask = get_sequence_mask(post_len)
        if is_training:
            logits = self.decoder(
                encoder_hidden,
                target=response_token,
                attn_value=encoder_outputs,
                attn_mask=encoder_outputs_mask,
                teaching_force_rate=self.teaching_force_rate,
                is_training=is_training
            )
        else:
            # test
            logits = self.decoder(
                encoder_hidden,
                target=None,
                attn_value=encoder_outputs,
                attn_mask=encoder_outputs_mask,
                is_training=False
            )
        outputs = Pack(logits=logits)
        return outputs

    def beam_search(self, inputs, beam_size=4, per_node_beam_size=4):
        # designed for test or interactive mode
        post_token, post_len = inputs.post
        embedded_post = self.post_embedding(post_token)
        encoder_outputs, encoder_hidden = self.encoder((embedded_post, post_len))
        encoder_outputs_mask = get_sequence_mask(post_len)
        all_top_k_predictions, log_probabilities = \
            self.decoder.forward_beam_search(encoder_hidden,
                                             attn_value=encoder_outputs,
                                             attn_mask=encoder_outputs_mask,
                                             beam_size=beam_size,
                                             per_node_beam_size=per_node_beam_size)
        predictions = all_top_k_predictions[:, 0, :]
        outputs = Pack(predictions=predictions)
        return outputs

    def collect_metrics(self, outputs, target):
        """
        collect_metrics
        """
        num_samples = target.size(0)
        metrics = Pack(num_samples=num_samples)

        logits = outputs.logits
        nll = self.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        predictions = logits.argmax(dim=2)
        acc = accuracy(predictions, target, padding_idx=self.padding_index, end_idx=self.end_index)
        metrics.add(nll=nll, acc=acc)
        metrics.add(ppx=math.exp(nll.item()))
        metrics.add(loss=nll)
        return metrics

    def iterate(self, inputs, optimizer=None, grad_clip=None):
        """
        iterate
        """
        outputs = self.forward(inputs, is_training=True)
        response_token, response_len = inputs.response
        target = response_token[:, 1:]
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
