from modules.transformer import TransformerEncoder
from modules.transformer import TransformerDecoder
from modules.utils import *
from models.base_model import BaseModel
from utils.pack import Pack
from utils.metrics import accuracy, perplexity
from torch.nn.utils import clip_grad_norm_
from modules.criterions import SequenceCrossEntropy
import torch.nn as nn
from overrides import overrides


class Transformer(BaseModel):
    def __init__(
            self,
            post_embedding,
            response_embedding,
            embedding_size,
            hidden_size,
            vocab_size,
            start_index,
            end_index,
            padding_index,
            num_heads,
            num_layers=2,
            dropout=0.2,
            learning_position_embedding=False,
            embedding_scale=False,
            num_positions=1024,
    ):
        super().__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.start_index = start_index
        self.end_index = end_index
        self.padding_index = padding_index
        self.dropout = dropout
        self.num_layers = num_layers

        self.encoder = TransformerEncoder(
            num_heads,
            num_layers,
            embedding_size,
            post_embedding,
            hidden_size,
            dropout=dropout,
            learn_position_embedding=learning_position_embedding,
            embedding_scale=embedding_scale,
            num_positions=num_positions
        )

        output_layer = nn.Sequential(
            nn.LayerNorm(embedding_size),
            nn.Linear(embedding_size, vocab_size)
        )

        self.decoder = TransformerDecoder(
            num_heads,
            num_layers,
            embedding_size,
            hidden_size,
            response_embedding,
            start_index,
            end_index,
            output_layer,
            dropout=dropout,
            embedding_scale=embedding_scale,
            learn_positional_embedding=learning_position_embedding,
            num_positions=num_positions
        )

    def forward(self, inputs, num_steps=50, is_training=True):
        """
        train and eval
        """
        if is_training:
            assert hasattr(inputs, 'response')
            if isinstance(inputs.response, tuple):
                response_tokens, response_len = inputs.response
            else:
                response_tokens = inputs.response
        if isinstance(inputs.post, tuple):
            post_tokens, post_len = inputs.post
        else:
            post_tokens = inputs.post
        encoder_mask = post_tokens.ne(self.padding_index)
        encoder_output = self.encoder(post_tokens, encoder_mask)
        if is_training:
            logits = self.decoder(
                encoder_output,
                encoder_mask,
                target=response_tokens,
                is_training=True
            )
        else:
            # test
            logits = self.decoder(
                encoder_output,
                encoder_mask,
                num_steps=num_steps,
                is_training=False
            )
        outputs = Pack(logits=logits)
        return outputs

    def beam_forward(self, inputs, beam_size=4, per_node_beam_size=4, num_steps=50):
        # designed for test or interactive mode
        if isinstance(inputs.post, tuple):
            post_tokens, post_len = inputs.post
        else:
            post_tokens = inputs.post
        encoder_mask = post_tokens.ne(self.padding_index)
        encoder_output = self.encoder(post_tokens, encoder_mask)
        all_top_k_predictions, log_probabilities = \
            self.decoder.beam_forward(
                encoder_output,
                encoder_mask,
                num_steps=num_steps,
                beam_size=beam_size,
                per_node_beam_size=per_node_beam_size
            )
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
        nll = sequence_cross_entropy_with_logits(logits, target, weights=target.ne(self.padding_index))
        loss += nll
        prediction = logits.argmax(dim=2)
        acc = accuracy(prediction, target, padding_idx=self.padding_index, end_idx=self.end_index, reduction="batch")
        metrics.add(nll=nll, acc=acc)
        metrics.add(ppx=math.exp(nll.item()))
        metrics.add(loss=loss)
        return metrics

    @overrides
    def iterate(self, inputs, optimizer=None, grad_clip=None):
        """
        iterate
        """
        outputs = self.forward(inputs, is_training=True)
        if isinstance(inputs.response, tuple):
            response_tokens, response_len = inputs.response
        else:
            response_tokens = inputs.response

        target = response_tokens[:, 1:].contiguous()
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
