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


class ProfileSeq2Seq(BaseModel):
    def __init__(
            self,
            post_embedding,
            response_embedding,
            profile_embedding,
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
        self.profile_embedding = profile_embedding
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
        num_classes = response_embedding.weight.size(0)
        self.decoder = ProfileDecoder(
            decoder_input_size,
            hidden_size,
            num_classes,
            start_index,
            end_index,
            response_embedding,
            profile_embedding,
            num_layers=num_layers,
            attention=decoder_attn,
            dropout=dropout
        )
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=padding_index, reduction='sum')
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
            response_token, response_len = inputs.response

        post_token, post_len = inputs.post
        embedded_post = self.post_embedding(post_token)
        encoder_outputs, encoder_hidden = self.encoder((embedded_post, post_len))
        encoder_outputs_mask = get_sequence_mask(post_len)
        if is_training:
            logits = self.decoder(
                encoder_hidden,
                inputs.speaker,
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
                inputs.speaker,
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
        num_tokens = target.ne(self.padding_index).sum().item()
        mean_nll = nll / num_tokens
        predictions = logits.argmax(dim=2)
        acc = accuracy(predictions, target, padding_idx=self.padding_index, end_idx=self.end_index)
        metrics.add(nll=mean_nll, acc=acc)
        metrics.add(ppx=math.exp(mean_nll.item()))
        if self.mmi_anti:
            gamma_logits = logits[:, :self.anti_gamma, :]
            gamma_target = target[:, :self.anti_gamma]
            loss -= self.anti_rate * \
                    self.cross_entropy(gamma_logits.reshape(-1, logits.size(-1)),
                                       gamma_target.reshape(-1))
        loss = loss / num_tokens
        metrics.add(loss=loss)
        return metrics

    @overrides
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


class ProfileDecoder(nn.Module):

    def __init__(self,
                 input_size,
                 hidden_size,
                 num_classes,
                 start_index,
                 end_index,
                 text_embedding,
                 profile_embedding,
                 num_layers=2,
                 attention=None,
                 dropout=0.0):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.start_index = start_index
        self.end_index = end_index
        self.text_embedding = text_embedding
        self.profile_embedding = profile_embedding
        self.attention = attention
        self.dropout = dropout
        self.num_layers = num_layers

        self.rnn = nn.GRU(input_size=self.input_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          batch_first=True,
                          dropout=dropout if self.num_layers > 1 else 0)

        self.output_layer = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_size, self.num_classes)
        )

    def forward(self,
                hidden,
                profile,
                target=None,
                attn_value=None,
                attn_mask=None,
                num_steps=50,
                teaching_force_rate=0.0,
                is_training=True):
        """
        forward

        :param
        hidden : ``torch.FloatTensor``, required.
            Initial hidden tensor of shape (num_layers, batch_size, hidden_size)
        profile : ``torch.LongTensor``, required.
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
            A ``torch.FloatTensor`` of shape (batch_size, num_steps, num_classes)
        """
        if self.attention is not None:
            assert attn_value is not None

        if target is not None:
            num_steps = target.size(1) - 1

        embedded_profile = self.profile_embedding(profile)
        last_predictions = hidden.new_full((hidden.size(1),), fill_value=self.start_index).long()
        step_logits = []
        for timestep in range(num_steps):
            if not is_training and (last_predictions == self.end_index).all():
                break
            if is_training and torch.rand(1).item() < teaching_force_rate:
                inputs = target[:, timestep]
            else:
                inputs = last_predictions
            # `outputs` of shape (batch_size, num_classes)
            outputs, hidden = self._take_step(inputs, hidden, embedded_profile, attn_value, attn_mask)
            # shape: (batch_size,)
            last_predictions = torch.argmax(outputs, dim=-1)
            step_logits.append(outputs.unsqueeze(1))

        logits = torch.cat(step_logits, dim=1)
        return logits

    def _take_step(self, inputs, hidden, profile, attn_value=None, attn_mask=None):
        # `inputs` of shape: (batch_size,)
        # `hidden` of shape: (num_layers, batch_size, hidden_size)
        # shape: (batch_size, input_size)
        embedded_inputs = self.text_embedding(inputs)
        rnn_inputs = embedded_inputs
        if self.attention is not None:
            # shape: (batch_size, num_rows)
            attn_score = self.attention(hidden[-1], attn_value, attn_mask)
            attn_inputs = torch.sum(attn_score.unsqueeze(2) * attn_value, dim=1)
            # shape: (batch_size, input_size + attn_size)
            rnn_inputs = torch.cat([rnn_inputs, attn_inputs], dim=-1)
        rnn_inputs = torch.cat([rnn_inputs, profile], dim=-1)
        _, next_hidden = self.rnn(rnn_inputs.unsqueeze(1), hidden)
        # shape: (batch_size, num_classes)
        outputs = self.output_layer(next_hidden[-1])
        return outputs, next_hidden

    def forward_beam_search(self,
                            hidden,
                            profile,
                            attn_value=None,
                            attn_mask=None,
                            num_steps=50,
                            beam_size=4,
                            per_node_beam_size=4):
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

        if self.attention is not None:
            assert attn_value is not None

        beam_search = BeamSearch(self.end_index, num_steps, beam_size, per_node_beam_size)
        start_predictions = hidden.new_full((hidden.size(1),), fill_value=self.start_index).long()
        embedded_profile = self.profile_embedding(profile)
        # `hidden` of shape: (batch_size, num_layers, hidden_size)
        hidden = hidden.transpose(0, 1).contiguous()
        state = {'hidden': hidden, 'profile': embedded_profile}
        if self.attention:
            state['attn_value'] = attn_value
            state['attn_mask'] = attn_mask
        all_top_k_predictions, log_probabilities = \
            beam_search.search(start_predictions, state, self._beam_step, early_stop=True)
        return all_top_k_predictions, log_probabilities

    def _beam_step(self, inputs, state):
        # shape: (group_size, input_size)
        embedded_inputs = self.text_embedding(inputs)
        rnn_inputs = embedded_inputs
        profile = state['profile']
        # shape: (group_size, num_layers, input_size)
        hidden = state['hidden']
        # shape: (num_layers, group_size, input_size)
        hidden = hidden.transpose(0, 1).contiguous()
        if self.attention is not None:
            attn_value = state['attn_value']
            attn_mask = state['attn_mask']
            # shape: (group_size, num_rows)
            attn_score = self.attention(hidden[-1], attn_value, attn_mask)
            attn_inputs = torch.sum(attn_score.unsqueeze(2) * attn_value, dim=1)
            # shape: (group_size, input_size + attn_size)
            rnn_inputs = torch.cat([rnn_inputs, attn_inputs], dim=-1)
        rnn_inputs = torch.cat([rnn_inputs, profile], dim=-1)
        _, next_hidden = self.rnn(rnn_inputs.unsqueeze(1), hidden)
        state['hidden'] = next_hidden.transpose(0, 1).contiguous()
        # shape: (group_size, num_classes)
        log_prob = F.log_softmax(self.output_layer(next_hidden[-1]), dim=-1)
        return log_prob, state
