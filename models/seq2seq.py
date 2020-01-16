from modules.rnn import GRUEncoder
from modules.rnn import StackGRUDecoder
from modules.attention import MLPAttention
from modules.utils import *
from models.base_model import BaseModel
from utils.pack import Pack
from utils.metrics import accuracy, perplexity
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn
from overrides import overrides


class Seq2Seq(BaseModel):
    def __init__(
            self,
            src_embedding,
            tgt_embedding,
            embedding_size,
            hidden_size,
            vocab_size,
            start_index,
            end_index,
            padding_index,
            bidirectional=True,
            num_layers=2,
            dropout=0.2,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.start_index = start_index
        self.end_index = end_index
        self.padding_index = padding_index
        self.dropout = dropout
        self.num_layers = num_layers
        if bidirectional:
            encoder_hidden_size = hidden_size//2
        else:
            encoder_hidden_size = hidden_size
        self.encoder = GRUEncoder(
            embedding_size,
            encoder_hidden_size,
            src_embedding,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout
        )
        output_layer = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(embedding_size, vocab_size)
        )
        decoder_attn = MLPAttention(hidden_size, hidden_size, hidden_size)
        decoder_input_size = embedding_size + hidden_size
        self.decoder = StackGRUDecoder(
            decoder_input_size,
            hidden_size,
            start_index,
            end_index,
            tgt_embedding,
            output_layer,
            num_layers=num_layers,
            attention=decoder_attn,
            dropout=dropout
        )

    def forward(self, inputs, num_steps=50, is_training=True):
        """
        train and eval
        """
        if is_training:
            assert hasattr(inputs, 'tgt')
            if isinstance(inputs.tgt, tuple):
                tgt_tokens, tgt_len = inputs.tgt
            else:
                tgt_tokens = inputs.tgt
        src_tokens, src_len = inputs.src
        encoder_output, encoder_hidden = self.encoder(inputs.src)
        encoder_mask = src_tokens.ne(self.padding_index)
        if is_training:
            logits = self.decoder(
                encoder_hidden,
                target=tgt_tokens,
                encoder_output=encoder_output,
                encoder_mask=encoder_mask,
                is_training=True
            )
        else:
            # test
            logits = self.decoder(
                encoder_hidden,
                encoder_output=encoder_output,
                encoder_mask=encoder_mask,
                is_training=False,
                num_steps=num_steps
            )
        outputs = Pack(logits=logits)
        return outputs

    def beam_forward(self, inputs, beam_size=4, per_node_beam_size=4, num_steps=50):
        # designed for test or interactive mode
        src_tokens, src_len = inputs.src
        encoder_output, encoder_hidden = self.encoder(inputs.src)
        encoder_mask = src_tokens.ne(self.padding_index)
        all_top_k_predictions, log_probabilities = \
            self.decoder.beam_forward(
                encoder_hidden,
                encoder_output=encoder_output,
                encoder_mask=encoder_mask,
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
        nll = sequence_cross_entropy_with_logits(logits, target, target.ne(self.padding_index))
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
        if isinstance(inputs.tgt, tuple):
            tgt_tokens, tgt_len = inputs.tgt
        else:
            tgt_tokens = inputs.tgt
        target = tgt_tokens[:, 1:].contiguous()
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
