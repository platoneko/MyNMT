from modules.rnn import GRUEncoder
from modules.memory_network import MemoryNetwork
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


def get_sentence_length(tokens, end_index):
    # `tokens` of shape: (batch_size, num_steps)
    batch_size, num_steps = tokens.size()
    tokens = torch.cat([tokens, tokens.new_full((batch_size, 1), fill_value=end_index)], dim=1)
    length = -((tokens == end_index).flip(1)).argmax(dim=1) + num_steps
    return length


class StyleSeq2Seq(BaseModel):
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
            ir,
            num_layers=2,
            dropout=0.2,
            teaching_force_rate=0.5,
            num_hops=2,
            topk=20,
            reinforce=True,
            reinforce_rate=0.5
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
        self.topk = topk

        self.encoder = GRUEncoder(embedding_size, embedding_size//2,
                                  dropout=dropout, num_layers=num_layers, bidirectional=True)

        context_attn = MLPAttention(hidden_size, embedding_size, hidden_size)
        style_attn = MLPAttention(hidden_size, embedding_size, hidden_size)

        vocab_size = response_embedding.weight.size(0)
        self.num_hops = num_hops
        self.memory_net = MemoryNetwork(num_hops, vocab_size, embedding_size, padding_index)

        decoder_input_size = embedding_size * 4
        self.decoder = StyleDecoder(
            input_size=decoder_input_size,
            hidden_size=hidden_size,
            embedding_size=embedding_size,
            vocab_size=vocab_size,
            start_index=start_index,
            end_index=end_index,
            text_embedding=response_embedding,
            profile_embedding=profile_embedding,
            context_attn=context_attn,
            style_attn=style_attn,
            num_layers=num_layers,
            dropout=dropout
        )
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=padding_index)
        self.reinforce = reinforce
        self.reinforce_rate = reinforce_rate

        self.ir = ir

    def forward(self, inputs, is_training=True):
        """
        train and eval
        """
        if is_training:
            assert inputs.response is not None
        if hasattr(inputs, 'response'):
            response_tokens, response_len = inputs.response

        post_tokens, post_len = inputs.post
        embedded_post = self.post_embedding(post_tokens)
        encoder_output, encoder_hidden = self.encoder((embedded_post, post_len))
        encoder_output_mask = post_tokens.ne(self.padding_index)

        # shape: (batch_size, k, length)
        #3if is_training:
        candidate = self.ir.get_candidate(inputs.candidate)
        # else:
        #    candidate = self.ir.search_candidate(post_tokens, self.topk, post_mask=encoder_output_mask)
        # shape: (batch_size, embedding_size)
        candidate_memory = self.memory_net(encoder_hidden[-1], candidate)
        # shape: (batch_size, k, embedding_size)
        style_memory = self.ir.get_style(inputs.speaker, self.topk)
        if is_training:
            logits = self.decoder(
                encoder_hidden,
                inputs.speaker,
                candidate_memory,
                style_memory,
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
                candidate_memory,
                style_memory,
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

        candidate = self.ir.get_candidate(inputs.candidate)
        # candidate = self.ir.search_candidate(post_tokens, self.topk, encoder_output_mask)
        candidate_memory = self.memory_net(encoder_hidden[-1], candidate)
        style_memory = self.ir.get_style(inputs.speaker, self.topk)

        all_top_k_predictions, log_probabilities = \
            self.decoder.forward_beam_search(
                encoder_hidden,
                inputs.speaker,
                candidate_memory,
                style_memory,
                attn_value=encoder_output,
                attn_mask=encoder_output_mask,
                beam_size=beam_size,
                per_node_beam_size=per_node_beam_size)
        prediction = all_top_k_predictions[:, 0, :]
        outputs = Pack(prediction=prediction)
        return outputs

    @overrides
    def collect_metrics(self, outputs, target, speaker=None):
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

        if self.reinforce:
            predict_length = get_sentence_length(prediction, self.end_index)
            predict_mask = get_sequence_mask(predict_length, prediction.size(1))
            predict_score = self.ir.get_style_score(prediction, predict_mask, speaker)
            target_mask = target.ne(self.padding_index) & target.ne(self.end_index)
            target_score = self.ir.get_style_score(target, target_mask, speaker)
            # shape: (batch_size,)
            reward = predict_score - target_score
            # shape: (batch_size * length)
            distribution = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                           prediction.reshape(-1),
                                           reduction='none')
            loss -= distribution.view(num_samples, -1) * \
                predict_mask.float() * \
                reward.unsqueeze(1) * \
                self.reinforce_rate
            metrics.add(reward=reward.mean())

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
        metrics = self.collect_metrics(outputs, target, inputs.speaker)

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


class StyleDecoder(nn.Module):

    def __init__(
            self,
            input_size,
            hidden_size,
            embedding_size,
            vocab_size,
            start_index,
            end_index,
            text_embedding,
            profile_embedding,
            context_attn,
            style_attn,
            num_layers=2,
            dropout=0.0
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.start_index = start_index
        self.end_index = end_index
        self.text_embedding = text_embedding
        self.profile_embedding = profile_embedding
        self.context_attn = context_attn
        self.style_attn = style_attn
        self.dropout = dropout
        self.num_layers = num_layers

        self.rnn = nn.GRU(input_size=self.input_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          batch_first=True,
                          dropout=dropout if self.num_layers > 1 else 0)

        self.output_layer = StyleAwareBias(embedding_size, hidden_size, vocab_size, dropout)

    def forward(
            self,
            hidden,
            profile,
            candidate_memory,
            style_memory,
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
        profile : ``torch.LongTensor``, required.
            Speaker tokens tensor of shape (batch_size,)
        candidate_memory : ``torch.FloatTensor``, required.
            Tensor of shape (batch_size, k, embedding_size)
        style_memory : ``torch.FloatTensor``, required.
            Tensor of shape (batch_size, k, embedding_size)
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
        if target is not None:
            num_steps = target.size(1) - 1

        embedded_profile = self.profile_embedding(profile)
        last_prediction = hidden.new_full((hidden.size(1),), fill_value=self.start_index).long()
        step_logits = []
        for timestep in range(num_steps):
            if not is_training and (last_prediction == self.end_index).all():
                break
            if is_training and torch.rand(1).item() < teaching_force_rate:
                input = target[:, timestep]
            else:
                input = last_prediction
            # `outputs` of shape (batch_size, vocab_size)
            outputs, hidden = self._take_step(
                input, hidden, embedded_profile,
                candidate_memory, style_memory,
                attn_value, attn_mask)
            # shape: (batch_size,)
            last_prediction = torch.argmax(outputs, dim=-1)
            step_logits.append(outputs.unsqueeze(1))

        logits = torch.cat(step_logits, dim=1)
        return logits

    def _take_step(
            self,
            input,
            hidden,
            profile,
            candidate_memory,
            style_memory,
            attn_value=None,
            attn_mask=None
    ):
        # `inputs` of shape: (batch_size,)
        # `hidden` of shape: (num_layers, batch_size, hidden_size)
        # shape: (batch_size, input_size)
        embedded_input = self.text_embedding(input)
        # shape: (batch_size, num_rows)
        context_score = self.context_attn(hidden[-1], attn_value, attn_mask)
        context_input = context_score.unsqueeze(1).matmul(attn_value).squeeze(1)
        rnn_input = torch.cat([embedded_input, context_input, profile, candidate_memory], dim=-1)
        _, next_hidden = self.rnn(rnn_input.unsqueeze(1), hidden)

        style_score = self.context_attn(hidden[-1], style_memory)
        style_input = style_score.unsqueeze(1).matmul(style_memory).squeeze(1)
        # shape: (batch_size, vocab_size)
        outputs = self.output_layer(next_hidden[-1], style_input)
        return outputs, next_hidden

    def forward_beam_search(self,
                            hidden,
                            profile,
                            candidate_memory,
                            style_memory,
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
        profile : ``torch.LongTensor``, required.
            Speaker tokens tensor of shape (batch_size,)
        candidate_memory : ``torch.FloatTensor``, required.
            Tensor of shape (batch_size, k, embedding_size)
        style_memory : ``torch.FloatTensor``, required.
            Tensor of shape (batch_size, k, embedding_size)
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
        beam_search = BeamSearch(self.end_index, num_steps, beam_size, per_node_beam_size)
        start_prediction = hidden.new_full((hidden.size(1),), fill_value=self.start_index).long()
        embedded_profile = self.profile_embedding(profile)
        # `hidden` of shape: (batch_size, num_layers, hidden_size)
        hidden = hidden.transpose(0, 1).contiguous()
        state = {'hidden': hidden,
                 'profile': embedded_profile,
                 'attn_value': attn_value,
                 'attn_mask': attn_mask,
                 'candidate_memory': candidate_memory,
                 'style_memory': style_memory}
        all_top_k_predictions, log_probabilities = \
            beam_search.search(start_prediction, state, self._beam_step, early_stop=True)
        return all_top_k_predictions, log_probabilities

    def _beam_step(self, input, state):
        # shape: (group_size, input_size)
        embedded_input = self.text_embedding(input)
        profile = state['profile']
        # shape: (group_size, num_layers, input_size)
        hidden = state['hidden']
        # shape: (num_layers, group_size, input_size)
        hidden = hidden.transpose(0, 1).contiguous()
        attn_value = state['attn_value']
        attn_mask = state['attn_mask']
        # shape: (group_size, num_rows)
        context_score = self.context_attn(hidden[-1], attn_value, attn_mask)
        context_input = torch.sum(context_score.unsqueeze(2) * attn_value, dim=1)
        candidate_memory = state['candidate_memory']
        # shape: (group_size, rnn_input_size)
        rnn_input = torch.cat([embedded_input, context_input, profile, candidate_memory], dim=-1)
        _, next_hidden = self.rnn(rnn_input.unsqueeze(1), hidden)
        state['hidden'] = next_hidden.transpose(0, 1).contiguous()

        style_memory = state['style_memory']
        style_score = self.context_attn(hidden[-1], style_memory)
        style_input = torch.sum(style_score.unsqueeze(2) * style_memory, dim=1)
        # shape: (group_size, vocab_size)
        log_prob = F.log_softmax(self.output_layer(next_hidden[-1], style_input), dim=-1)
        return log_prob, state


class StyleAwareBias(nn.Module):
    def __init__(self,
                 embedding_size,
                 state_size,
                 vocab_size,
                 dropout=0.0):
        super().__init__()
        self.linear_i = nn.Linear(embedding_size, vocab_size, bias=False)
        self.linear_s = nn.Linear(state_size, vocab_size, bias=False)
        self.out_bias = nn.Parameter(torch.zeros(vocab_size))
        self.linear_o = nn.Linear(state_size, 1, bias=False)
        self.dropout = dropout

    def forward(self, state, input):
        # shape: (batch_size, 1)
        a_t = F.sigmoid(self.linear_o(state))
        output = (1 - a_t) * F.dropout(self.linear_i(input), p=self.dropout) + \
                  a_t * F.dropout(self.linear_s(state), p=self.dropout) + self.out_bias
        # shape: (batch_size, num_classes)
        return output


class InfoRetriever(object):

    def __init__(
            self,
            candidate_lib, style_lib,
            candidate_vectors=None,
            ranker_embedding=None,
            classifier=None
    ):
        # shape: (num_candidate, length)
        self.candidate_lib = candidate_lib
        # shape: (num_style, K, embedding_size)
        self.style_lib = style_lib
        # shape: (embedding_size, num_candidate)
        self.candidate_vectors = candidate_vectors
        self.ranker_embedding = ranker_embedding
        self.classifier = classifier
        if classifier is not None:
            self.classifier.eval()

    def get_candidate(self, candidate_indices):
        # `candidate_indices` of shape (batch_size, k)
        batch_size, k = candidate_indices.size()
        candidate = torch.index_select(self.candidate_lib, 0, candidate_indices.view(-1))
        candidate = candidate.view(batch_size, k, -1)
        return candidate

    def search_candidate(self, post_tokens, topk, post_mask=None):
        # `post_token` of shape: (batch_size, length)
        # `post_mask` of shape: (batch_size, length)
        embedded_post = self.ranker_embedding(post_tokens)
        # shape: (batch_size, embedding_size)
        if post_mask is not None:
            post_vector = masked_sum(embedded_post, post_mask.unsqueeze(2), 1)
        else:
            post_vector = torch.sum(embedded_post, 1)
        # shape: (batch_size, num_candidate)
        rank_score = post_vector.matmul(self.candidate_vectors)
        # shape: (batch_size, k)
        _, topk_indices = rank_score.topk(topk, 1)
        candidate = self.get_candidate(topk_indices)
        return candidate

    def get_style(self, indices, topk):
        # `indices` of shape (batch_size,)
        topk = min(self.style_lib.size(1), topk)
        return self.style_lib.index_select(0, indices)[:, :topk, :]

    def get_style_score(self, input, speaker, mask):
        # `inputs` of shape: (batch_size, length)
        with torch.no_grad():
            outputs = self.classifier(input, mask)
        style_score = outputs.probability[torch.arange(0, speaker.size(0)), speaker]
        return style_score
