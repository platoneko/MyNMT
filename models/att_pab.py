from modules.rnn import GRUEncoder
from modules.attention import MLPAttention
from models.base_model import BaseModel
from utils.pack import Pack
from utils.metrics import accuracy
from modules.criterions import SequenceCrossEntropy
from modules.beam_search import BeamSearch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_


class AttPAB(BaseModel):
    """
    Att.+PAB model in ``Personalized Dialogue Generation with Diversified Traits``
    """

    def __init__(
            self,
            text_embedding,
            loc_embedding,
            gender_embedding,
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
        self.text_embedding = text_embedding
        self.loc_embedding = loc_embedding
        self.gender_embedding = gender_embedding
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.start_index = start_index
        self.end_index = end_index
        self.padding_index = padding_index
        self.dropout = dropout
        self.teaching_force_rate = teaching_force_rate
        self.num_layers = num_layers

        self.encoder = GRUEncoder(embedding_size, hidden_size // 2, dropout=dropout, num_layers=num_layers)

        decoder_attn = MLPAttention(hidden_size, hidden_size, hidden_size)
        profile_attn = MLPAttention(hidden_size, embedding_size, hidden_size)
        tags_attn = MLPAttention(hidden_size, embedding_size, hidden_size)

        decoder_input_size = embedding_size + hidden_size
        num_classes = text_embedding.weight.size(0)
        self.decoder = AttPABDecoder(
            input_size=decoder_input_size,
            hidden_size=hidden_size,
            embedding_size=embedding_size,
            num_classes=num_classes,
            start_index=start_index,
            end_index=end_index,
            text_embedding=text_embedding,
            loc_embedding=loc_embedding,
            gender_embedding=gender_embedding,
            decoder_attn=decoder_attn,
            profile_attn=profile_attn,
            tags_attn=tags_attn,
            num_layers=num_layers,
            dropout=dropout
        )
        weight = torch.ones(num_classes)
        weight[padding_index] = 0.0
        self.cross_entropy = SequenceCrossEntropy(padding_idx=padding_index, weight=weight)

    def forward(self, inputs, evaluation=False):
        """
        train and eval
        """
        if self.training or evaluation:
            assert inputs.response is not None
        if hasattr(inputs, 'response'):
            response_token, response_len = inputs.response

        post_token, post_len = inputs.post
        embedded_post = self.text_embedding(post_token)

        encoder_outputs, encoder_hidden = self.encoder((embedded_post, post_len))
        encoder_outputs_mask = post_token.ne(self.padding_index)

        tags_token, _, tags_length = inputs.tags

        if self.training:
            logits = self.decoder(
                encoder_hidden,
                loc=inputs.loc,
                gender=inputs.gender,
                tags=(tags_token, tags_length),
                response=response_token,
                attn_value=encoder_outputs,
                attn_mask=encoder_outputs_mask,
                teaching_force_rate=self.teaching_force_rate
            )
        elif evaluation:
            # eval, we need to obtain targets max len, so `target` is required
            logits = self.decoder(
                encoder_hidden,
                loc=inputs.loc,
                gender=inputs.gender,
                tags=(tags_token, tags_length),
                response=response_token,
                attn_value=encoder_outputs,
                attn_mask=encoder_outputs_mask
            )
        else:
            # test
            logits = self.decoder(
                encoder_hidden,
                loc=inputs.loc,
                gender=inputs.gender,
                tags=(tags_token, tags_length),
                response=None,
                attn_value=encoder_outputs,
                attn_mask=encoder_outputs_mask,
                early_stop=True
            )
        outputs = Pack(logits=logits)
        return outputs

    def beam_search(self, inputs, beam_size=4, per_node_beam_size=4):
        # designed for test or interactive mode
        post_token, post_len = inputs.post
        embedded_post = self.text_embedding(post_token)
        encoder_outputs, encoder_hidden = self.encoder((embedded_post, post_len))
        encoder_outputs_mask = post_token.ne(self.padding_index)

        tags_token, _, tags_length = inputs.tags

        all_top_k_predictions, log_probabilities = \
            self.decoder.forward_beam_search(encoder_hidden,
                                             loc=inputs.loc,
                                             gender=inputs.gender,
                                             tags=(tags_token, tags_length),
                                             attn_value=encoder_outputs,
                                             attn_mask=encoder_outputs_mask,
                                             beam_size=beam_size,
                                             per_node_beam_size=per_node_beam_size,
                                             early_stop=True)
        predictions = all_top_k_predictions[:, 0, :]
        outputs = Pack(predictions=predictions)
        return outputs

    def collect_metrics(self, outputs, target):
        """
        collect_metrics
        """
        num_samples = target.size(0)
        metrics = Pack(num_samples=num_samples)
        loss = 0

        logits = outputs.logits
        nll = self.cross_entropy(logits, target)
        predictions = logits.argmax(dim=2)
        acc = accuracy(predictions, target, padding_idx=self.padding_index)
        ppl = nll.exp()
        metrics.add(nll=nll, acc=acc, ppl=ppl)
        loss += nll

        metrics.add(loss=loss)
        return metrics

    def iterate(self, inputs, optimizer=None, grad_clip=None):
        """
        iterate
        """
        outputs = self.forward(inputs, evaluation=not self.training)
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


class AttPABDecoder(nn.Module):
    """
    Att.+PAB decoder in ``Personalized Dialogue Generation with Diversified Traits``
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 embedding_size,
                 num_classes,
                 start_index,
                 end_index,
                 text_embedding,
                 loc_embedding,
                 gender_embedding,
                 decoder_attn,
                 profile_attn,
                 tags_attn,
                 num_layers=2,
                 dropout=0.0):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.start_index = start_index
        self.end_index = end_index

        self.text_embedding = text_embedding
        self.loc_embedding = loc_embedding
        self.gender_embedding = gender_embedding

        self.decoder_attn = decoder_attn
        self.profile_attn = profile_attn
        self.tags_attn = tags_attn

        self.dropout = dropout
        self.num_layers = num_layers

        self.rnn = nn.GRU(input_size=self.input_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          batch_first=True,
                          dropout=dropout if self.num_layers > 1 else 0)

        self.tags_encoder = GRUEncoder(embedding_size, embedding_size, bidirectional=False)

        self.output_layer = PersonaAwareBias(embedding_size, hidden_size, num_classes)

    def forward(self, hidden,
                loc, gender, tags,
                attn_value,
                attn_mask,
                response=None,
                num_steps=50,
                teaching_force_rate=0.0,
                early_stop=False):
        """
        forward

        :param
        hidden : ``torch.FloatTensor``, required.
            Initial hidden tensor of shape (num_layers, batch_size, hidden_size)
        loc : ``torch.LongTensor``, required.
            Location tokens tensor of shape (batch_size,)
        gender : ``torch.LongTensor``, required.
            Gender tokens tensor of shape (batch_size,)
        tags : ``Tuple(torch.LongTensor, torch.LongTensor)``, required.
            Tag tokens tensor of shape (batch_size, num_tags, tag_len) with
            length tensor of shape (batch_size, num_tags)
        response : ``torch.LongTensor``, optional (default = None)
            Response tokens tensor of shape (batch_size, length)
        attn_value : ``torch.FloatTensor``, required.
            A ``torch.FloatTensor`` of shape (batch_size, num_rows, value_size)
        attn_mask : ``torch.LongTensor``, required.
            A ``torch.LongTensor`` of shape (batch_size, num_rows)
        teaching_force_rate : ``float``, optional (default = 0.0)
        early_stop : ``bool``, optional (default = False).
            If every predicted token from the last step is `self.end_index`, then we can stop early.

        :return
        logits : ``torch.FloatTensor``
            A ``torch.FloatTensor`` of shape (batch_size, num_steps, num_classes)
        """
        if response is not None:
            num_steps = response.size(1) - 1

        # shape: (batch_size, embedding_size)
        embedded_loc = self.loc_embedding(loc)
        # shape: (batch_size, embedding_size)
        embedded_gender = self.gender_embedding(gender)

        tags_token, tags_length = tags
        batch_size, num_tags = tags_length.size()
        # shape: (batch_size, num_tags, tag_len, embedding_size)
        embedded_tags = self.text_embedding(tags_token)
        # shape: (1, batch_size * num_tags, embedding_size)
        _, encoded_tags = self.tags_encoder((embedded_tags.view(batch_size * num_tags, -1, self.embedding_size),
                                             tags_length.view(-1)))
        encoded_tags = encoded_tags.view(batch_size, num_tags, self.embedding_size)
        tags_mask = tags_length.ne(0)

        last_predictions = hidden.new_full((hidden.size(1),), fill_value=self.start_index).long()
        step_logits = []
        for timestep in range(num_steps):
            if early_stop and (last_predictions == self.end_index).all():
                break
            if self.training and torch.rand(1).item() < teaching_force_rate:
                last_predictions = response[:, timestep]
            # `outputs` of shape (batch_size, num_classes)
            outputs, hidden = self._take_step(last_predictions,
                                              hidden,
                                              loc=embedded_loc,
                                              gender=embedded_gender,
                                              tags=(encoded_tags, tags_mask),
                                              attn_value=attn_value,
                                              attn_mask=attn_mask)
            # shape: (batch_size,)
            last_predictions = torch.argmax(outputs, dim=-1)
            step_logits.append(outputs.unsqueeze(1))

        logits = torch.cat(step_logits, dim=1)
        return logits

    def _take_step(self, inputs, hidden, loc, gender, tags,
                   attn_value, attn_mask):
        # `inputs` of shape: (batch_size,)
        # `hidden` of shape: (num_layers, batch_size, hidden_size)
        # `loc` of shape: (batch_size, embedding_size)
        # `gender` of shape: (batch_size, embedding_size)
        # `tags`: Tuple(encoded_tags, tags_mask)
        # `encoded_tags` of shape: (batch_size, num_tags, embedding_size)
        # `tags_mask` of shape: (batch_size, num_tags)

        # shape: (batch_size, input_size)
        embedded_inputs = self.text_embedding(inputs)
        # shape: (batch_size, num_rows)
        decoder_attn_score = self.decoder_attn(hidden[-1], attn_value, attn_mask)
        decoder_attn_inputs = torch.sum(decoder_attn_score.unsqueeze(2) * attn_value, dim=1)

        encoded_tags, tags_mask = tags
        # shape: (batch_size, num_tags)
        tags_attn_score = self.tags_attn(hidden[-1], encoded_tags, tags_mask)
        # shape: (batch_size, embedding_size)
        weighted_tags = torch.sum(tags_attn_score.unsqueeze(2) * encoded_tags, dim=1)
        # shape: (batch_size, 3, embedding_size)
        profile = torch.cat([loc, gender, weighted_tags], dim=1).view(-1, 3, self.embedding_size)
        profile_attn_score = self.profile_attn(hidden[-1], profile)
        weighted_profile = torch.sum(profile_attn_score.unsqueeze(2) * profile, dim=1)

        # shape: (batch_size, input_size + attn_size)
        rnn_inputs = torch.cat([embedded_inputs, decoder_attn_inputs], dim=-1)
        _, next_hidden = self.rnn(rnn_inputs.unsqueeze(1), hidden)
        # shape: (batch_size, num_classes)
        outputs = self.output_layer(next_hidden[-1], weighted_profile)
        return outputs, next_hidden

    def forward_beam_search(self,
                            hidden,
                            loc,
                            gender,
                            tags,
                            attn_value,
                            attn_mask,
                            num_steps=50,
                            beam_size=4,
                            per_node_beam_size=4,
                            early_stop=False):
        """
        Decoder forward using beam search at inference stage

        :param
        hidden : ``torch.FloatTensor``, required.
            Initial hidden tensor of shape (num_layers, batch_size, hidden_size)
        loc : ``torch.LongTensor``, required.
            Location tokens tensor of shape (batch_size,)
        gender : ``torch.LongTensor``, required.
            Gender tokens tensor of shape (batch_size,)
        tags : ``Tuple(torch.LongTensor, torch.LongTensor)``, required.
            Tag tokens tensor of shape (batch_size, num_tags, tag_len) with
            length tensor of shape (batch_size, num_tags)
        attn_value : ``torch.FloatTensor``, required.
            A ``torch.FloatTensor`` of shape (batch_size, num_rows, value_size)
        attn_mask : ``torch.LongTensor``, required.
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

        beam_search = BeamSearch(self.end_index, num_steps, beam_size, per_node_beam_size)
        start_predictions = hidden.new_full((hidden.size(1),), fill_value=self.start_index).long()

        # shape: (batch_size, embedding_size)
        embedded_loc = self.loc_embedding(loc)
        # shape: (batch_size, embedding_size)
        embedded_gender = self.gender_embedding(gender)

        tags_token, tags_length = tags
        batch_size, num_tags = tags_length.size()
        # shape: (batch_size, num_tags, tag_len, embedding_size)
        embedded_tags = self.text_embedding(tags_token)
        # shape: (1, batch_size * num_tags, embedding_size)
        _, encoded_tags = self.tags_encoder((embedded_tags.view(batch_size * num_tags, -1, self.embedding_size),
                                             tags_length.view(-1)))
        encoded_tags = encoded_tags.view(batch_size, num_tags, self.embedding_size)
        tags_mask = tags_length.ne(0)
        # `hidden` of shape: (batch_size, num_layers, hidden_size)
        hidden = hidden.transpose(0, 1).contiguous()
        state = {'hidden': hidden,
                 'attn_value': attn_value,
                 'attn_mask': attn_mask,
                 'loc': embedded_loc,
                 'gender': embedded_gender,
                 'encoded_tags': encoded_tags,
                 'tags_mask': tags_mask}
        all_top_k_predictions, log_probabilities = \
            beam_search.search(start_predictions, state, self._beam_step, early_stop=early_stop)
        return all_top_k_predictions, log_probabilities

    def _beam_step(self, inputs, state):
        # shape: (group_size, input_size)
        embedded_inputs = self.text_embedding(inputs)

        # shape: (group_size, num_layers, input_size)
        hidden = state['hidden']
        # shape: (num_layers, group_size, input_size)
        hidden = hidden.transpose(0, 1).contiguous()
        attn_value = state['attn_value']
        attn_mask = state['attn_mask']
        # shape: (group_size, num_rows)
        decoder_attn_score = self.decoder_attn(hidden[-1], attn_value, attn_mask)
        decoder_attn_inputs = torch.sum(decoder_attn_score.unsqueeze(2) * attn_value, dim=1)

        encoded_tags = state['encoded_tags']
        tags_mask = state['tags_mask']
        loc = state['loc']
        gender = state['gender']
        # shape: (group_size, num_tags)
        tags_attn_score = self.tags_attn(hidden[-1], encoded_tags, tags_mask)
        # shape: (group_size, embedding_size)
        weighted_tags = torch.sum(tags_attn_score.unsqueeze(2) * encoded_tags, dim=1)
        # shape: (group_size, 3, embedding_size)
        profile = torch.cat([loc, gender, weighted_tags], dim=1).view(-1, 3, self.embedding_size)
        profile_attn_score = self.profile_attn(hidden[-1], profile)
        weighted_profile = torch.sum(profile_attn_score.unsqueeze(2) * profile, dim=1)
        # shape: (group_size, input_size + attn_size)
        rnn_inputs = torch.cat([embedded_inputs, decoder_attn_inputs], dim=-1)
        _, next_hidden = self.rnn(rnn_inputs.unsqueeze(1), hidden)
        state['hidden'] = next_hidden.transpose(0, 1).contiguous()
        # shape: (group_size, num_classes)
        log_prob = F.log_softmax(self.output_layer(next_hidden[-1], weighted_profile), dim=-1)
        return log_prob, state


class PersonaAwareBias(nn.Module):
    def __init__(self,
                 profile_size,
                 state_size,
                 num_classes):
        super().__init__()
        self.linear_p = nn.Linear(profile_size, num_classes, bias=False)
        self.linear_s = nn.Linear(state_size, num_classes, bias=False)
        self.bias_o = nn.Parameter(torch.zeros(1, num_classes))
        self.linear_o = nn.Linear(state_size, 1, bias=False)

    def forward(self, state, profile):
        # shape: (batch_size, 1)
        a_t = F.sigmoid(self.linear_o(state))
        outputs = (1 - a_t) * self.linear_p(profile) + a_t * self.linear_s(state) + self.bias_o
        # shape: (batch_size, num_classes)
        return outputs
