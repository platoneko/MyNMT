from modules.rnn import GRUEncoder
from modules.attention import MLPAttention
from modules.beam_search import BeamSearch
from modules.utils import *
from models.base_model import BaseModel
from utils.pack import Pack
from utils.metrics import accuracy, perplexity

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from overrides import overrides


class SpeakerSeq2Seq(BaseModel):
    def __init__(
            self,
            post_embedding,
            response_embedding,
            speaker_embedding,
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
        self.speaker_embedding = speaker_embedding
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
        decoder_input_size = embedding_size + hidden_size + embedding_size
        vocab_size = response_embedding.weight.size(0)
        self.decoder = SpeakerDecoder(
            decoder_input_size,
            hidden_size,
            vocab_size,
            start_index,
            end_index,
            response_embedding,
            speaker_embedding,
            num_layers=num_layers,
            context_attn=decoder_attn,
            dropout=dropout
        )
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=padding_index, reduction='mean')

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
                inputs.speaker,
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
                inputs.speaker,
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
            self.decoder.forward_beam_search(
                encoder_hidden,
                inputs.speaker,
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
        prediction = logits.argmax(dim=2)
        acc = accuracy(prediction, target, padding_idx=self.padding_index, end_idx=self.end_index)
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


class SpeakerDecoder(nn.Module):

    def __init__(
            self,
            input_size,
            hidden_size,
            vocab_size,
            start_index,
            end_index,
            text_embedding,
            speaker_embedding,
            num_layers=2,
            context_attn=None,
            dropout=0.0
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.start_index = start_index
        self.end_index = end_index
        self.text_embedding = text_embedding
        self.speaker_embedding = speaker_embedding
        self.context_attn = context_attn
        self.dropout = dropout
        self.num_layers = num_layers

        self.rnn = nn.GRU(input_size=self.input_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          batch_first=True,
                          dropout=dropout if self.num_layers > 1 else 0)

        self.output_layer = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_size, self.vocab_size)
        )

    def forward(
            self,
            hidden,
            speaker,
            target=None,
            attn_value=None,
            attn_mask=None,
            num_steps=50,
            teaching_force_rate=0.0,
            is_training=True
    ):
        """
        forward

        :param
        hidden : ``torch.FloatTensor``, required.
            Initial hidden tensor of shape (num_layers, batch_size, hidden_size)
        speaker : ``torch.LongTensor``, required.
            Speaker tokens tensor of shape (batch_size,)
        target : ``torch.LongTensor``, optional (default = None)
            Target tokens tensor of shape (batch_size, length)
        attn_value : ``torch.FloatTensor``, optional (default = None)
            A ``torch.FloatTensor`` of shape (batch_size, num_rows, value_size)
        attn_mask : ``torch.LongTensor``, optional (default = None)
            A ``torch.LongTensor`` of shape (batch_size, num_rows)
        teaching_force_rate : ``float``, optional (default = 0.0)

        :return
        logits : ``torch.FloatTensor``
            A ``torch.FloatTensor`` of shape (batch_size, num_steps, vocab_size)
        """
        if self.context_attn is not None:
            assert attn_value is not None

        if target is not None:
            num_steps = target.size(1) - 1

        embedded_speaker = self.speaker_embedding(speaker)
        last_prediction = hidden.new_full((hidden.size(1),), fill_value=self.start_index).long()
        step_logits = []
        for timestep in range(num_steps):
            if not is_training and (last_prediction == self.end_index).all():
                break
            if is_training and torch.rand(1).item() < teaching_force_rate:
                input = target[:, timestep]
            else:
                input = last_prediction
            # `output` of shape (batch_size, vocab_size)
            output, hidden = self._take_step(input, hidden, embedded_speaker, attn_value, attn_mask)
            # shape: (batch_size,)
            last_prediction = torch.argmax(output, dim=-1)
            step_logits.append(output.unsqueeze(1))

        logits = torch.cat(step_logits, dim=1)
        return logits

    def _take_step(self, input, hidden, speaker, attn_value=None, attn_mask=None):
        # `input` of shape: (batch_size,)
        # `hidden` of shape: (num_layers, batch_size, hidden_size)
        # shape: (batch_size, input_size)
        embedded_input = self.text_embedding(input)
        rnn_input = embedded_input
        if self.context_attn is not None:
            # shape: (batch_size, num_rows)
            attn_score = self.context_attn(hidden[-1], attn_value, attn_mask)
            attn_input = attn_score.unsqueeze(1).matmul(attn_value).squeeze(1)
            # shape: (batch_size, input_size + attn_size)
            rnn_input = torch.cat([rnn_input, attn_input], dim=-1)
        rnn_input = torch.cat([rnn_input, speaker], dim=-1)
        _, next_hidden = self.rnn(rnn_input.unsqueeze(1), hidden)
        # shape: (batch_size, vocab_size)
        output = self.output_layer(next_hidden[-1])
        return output, next_hidden

    def forward_beam_search(
            self,
            hidden,
            speaker,
            attn_value=None,
            attn_mask=None,
            num_steps=50,
            beam_size=4,
            per_node_beam_size=4
    ):
        """
        Decoder forward using beam search at inference stage

        :param
        hidden : ``torch.FloatTensor``, required.
            Initial hidden tensor of shape (num_layers, batch_size, hidden_size)
        end_index : ``int``, required.
            Vocab index of <eos> token.
        attn_value : ``torch.FloatTensor``, optional (default = None)
            A ``torch.FloatTensor`` of shape (batch_size, num_rows, value_size)
        attn_mask : ``torch.LongTensor``, optional (default = None)
            A ``torch.LongTensor`` of shape (batch_size, num_rows)
        beam_size : ``int``, optional (default = 4)
        per_node_beam_size : ``int``, optional (default = 4)
        early_stop : ``bool``, optional (default = False).
            If every predicted token from the last step is `self.end_index`, then we can stop early.

        :return
        all_top_k_predictions : ``torch.LongTensor``
            A ``torch.LongTensor`` of shape (batch_size, beam_size, num_steps),
            containing k top sequences in descending order along dim 1.
        log_probabilities : ``torch.FloatTensor``
            A ``torch.FloatTensor``  of shape (batch_size, beam_size),
            Log probabilities of k top sequences.
        """

        if self.context_attn is not None:
            assert attn_value is not None

        beam_search = BeamSearch(self.end_index, num_steps, beam_size, per_node_beam_size)
        start_prediction = hidden.new_full((hidden.size(1),), fill_value=self.start_index).long()
        embedded_speaker = self.speaker_embedding(speaker)
        # `hidden` of shape: (batch_size, num_layers, hidden_size)
        hidden = hidden.transpose(0, 1).contiguous()
        state = {'hidden': hidden, 'speaker': embedded_speaker}
        if self.context_attn:
            state['attn_value'] = attn_value
            state['attn_mask'] = attn_mask
        all_top_k_predictions, log_probabilities = \
            beam_search.search(start_prediction, state, self._beam_step, early_stop=True)
        return all_top_k_predictions, log_probabilities

    def _beam_step(self, input, state):
        # shape: (group_size, input_size)
        embedded_input = self.text_embedding(input)
        rnn_input = embedded_input
        speaker = state['speaker']
        # shape: (group_size, num_layers, input_size)
        hidden = state['hidden']
        # shape: (num_layers, group_size, input_size)
        hidden = hidden.transpose(0, 1).contiguous()
        if self.context_attn is not None:
            attn_value = state['attn_value']
            attn_mask = state['attn_mask']
            # shape: (group_size, num_rows)
            attn_score = self.context_attn(hidden[-1], attn_value, attn_mask)
            attn_input = torch.sum(attn_score.unsqueeze(2) * attn_value, dim=1)
            # shape: (group_size, input_size + attn_size)
            rnn_input = torch.cat([rnn_input, attn_input], dim=-1)
        rnn_input = torch.cat([rnn_input, speaker], dim=-1)
        _, next_hidden = self.rnn(rnn_input.unsqueeze(1), hidden)
        state['hidden'] = next_hidden.transpose(0, 1).contiguous()
        # shape: (group_size, vocab_size)
        log_prob = F.log_softmax(self.output_layer(next_hidden[-1]), dim=-1)
        return log_prob, state
