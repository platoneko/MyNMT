from modules.rnn import GRUEncoder, StackGRUDecoder
from modules.attention import MLPAttention
from models.base_model import BaseModel
from modules.utils import *
from utils.pack import Pack
from utils.metrics import accuracy
from modules.criterions import SequenceCrossEntropy
from modules.beam_search import BeamSearch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_


class Redecode(BaseModel):
    """
    Att.+PAB model in ``Personalized Dialogue Generation with Diversified Traits`` with redecode framework
    """

    def __init__(
            self,
            text_embedding,
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
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.start_index = start_index
        self.end_index = end_index
        self.padding_index = padding_index
        self.dropout = dropout
        self.teaching_force_rate = teaching_force_rate
        self.num_layers = num_layers

        self.encoder = GRUEncoder(embedding_size, hidden_size // 2, dropout=dropout, num_layers=num_layers)

        decoder_1_attn = MLPAttention(hidden_size, hidden_size, hidden_size)
        decoder_2_attn = MLPAttention(hidden_size, hidden_size, hidden_size)

        num_classes = text_embedding.weight.size(0)
        self.decoder = RedecodeDecoder(
            input_size_1=embedding_size+hidden_size,
            input_size_2=embedding_size+hidden_size*2,
            hidden_size=hidden_size,
            embedding_size=embedding_size,
            num_classes=num_classes,
            start_index=start_index,
            end_index=end_index,
            text_embedding=text_embedding,
            decoder_1_attn=decoder_1_attn,
            decoder_2_attn=decoder_2_attn,
            text_encoder=self.encoder,
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

        if self.training:
            logits_1, logits_2 = self.decoder(
                encoder_hidden,
                response=response_token,
                attn_value=encoder_outputs,
                attn_mask=encoder_outputs_mask,
                teaching_force_rate=self.teaching_force_rate
            )
        elif evaluation:
            # eval, we need to obtain targets max len, so `target` is required
            logits_1, logits_2 = self.decoder(
                encoder_hidden,
                response=response_token,
                attn_value=encoder_outputs,
                attn_mask=encoder_outputs_mask
            )
        else:
            # test
            logits_1, logits_2 = self.decoder(
                encoder_hidden,
                response=None,
                attn_value=encoder_outputs,
                attn_mask=encoder_outputs_mask,
                early_stop=True
            )
        outputs = Pack(logits_1=logits_1, logits_2=logits_2)
        return outputs

    def beam_search(self, inputs, beam_size=4, per_node_beam_size=4):
        # designed for test or interactive mode
        post_token, post_len = inputs.post
        embedded_post = self.text_embedding(post_token)
        encoder_outputs, encoder_hidden = self.encoder((embedded_post, post_len))
        encoder_outputs_mask = post_token.ne(self.padding_index)

        predictions_1, predictions_2 = \
            self.decoder.forward_beam_search(encoder_hidden,
                                             attn_value=encoder_outputs,
                                             attn_mask=encoder_outputs_mask,
                                             beam_size=beam_size,
                                             per_node_beam_size=per_node_beam_size,
                                             early_stop=True)
        outputs = Pack(predictions_1=predictions_1, predictions_2=predictions_2)
        return outputs

    def collect_metrics(self, outputs, target):
        """
        collect_metrics
        """
        num_samples = target.size(0)
        metrics = Pack(num_samples=num_samples)
        loss = 0

        logits_1 = outputs.logits_1
        nll_1 = self.cross_entropy(logits_1, target)
        predictions_1 = logits_1.argmax(dim=2)
        acc_1 = accuracy(predictions_1, target, padding_idx=self.padding_index)
        ppl_1 = nll_1.exp()
        metrics.add(nll_1=nll_1, acc_1=acc_1, ppl_1=ppl_1)
        loss += nll_1

        logits_2 = outputs.logits_2
        nll_2 = self.cross_entropy(logits_2, target)
        predictions_2 = logits_2.argmax(dim=2)
        acc_2 = accuracy(predictions_2, target, padding_idx=self.padding_index)
        ppl_2 = nll_2.exp()
        metrics.add(nll_2=nll_2, acc_2=acc_2, ppl_2=ppl_2)
        loss += nll_2

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


class RedecodeDecoder(nn.Module):
    """
    Att.+PAB decoder in ``Personalized Dialogue Generation with Diversified Traits`` with redecode framework
    """
    def __init__(self,
                 input_size_1,
                 input_size_2,
                 hidden_size,
                 embedding_size,
                 num_classes,
                 start_index,
                 end_index,
                 text_embedding,
                 decoder_1_attn,
                 decoder_2_attn,
                 text_encoder,
                 num_layers=2,
                 dropout=0.0):
        super().__init__()

        self.input_size_1 = input_size_1
        self.input_size_2 = input_size_2

        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.start_index = start_index
        self.end_index = end_index

        self.text_embedding = text_embedding

        self.decoder_1_attn = decoder_1_attn
        self.decoder_2_attn = decoder_2_attn

        self.text_encoder = text_encoder

        self.dropout = dropout
        self.num_layers = num_layers

        self.decoder_1 = StackGRUDecoder(
            input_size=input_size_1,
            hidden_size=hidden_size,
            num_classes=num_classes,
            start_index=start_index,
            end_index=end_index,
            embedding=text_embedding,
            num_layers=num_layers,
            attention=decoder_1_attn,
            dropout=dropout
        )

        self.decoder_2 = nn.GRU(
            input_size=self.input_size_2,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout if self.num_layers > 1 else 0)

        self.output_layer = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_size, self.num_classes)
        )

    def forward(self, hidden,
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
        logits_1 : ``torch.FloatTensor``
            A ``torch.FloatTensor`` of shape (batch_size, num_steps, num_classes),
            first decoder outputs.
        logits_2 : ``torch.FloatTensor``
            A ``torch.FloatTensor`` of shape (batch_size, num_steps, num_classes),
            redecoder outputs.
        """
        if response is not None:
            num_steps = response.size(1) - 1

        # shape: (batch_size, num_steps, num_classes)
        logits_1 = self.decoder_1(
            hidden=hidden,
            target=response,
            attn_value=attn_value,
            attn_mask=attn_mask,
            num_steps=num_steps,
            teaching_force_rate=teaching_force_rate,
            early_stop=early_stop
        )

        # `pred_token` of shape: (batch_size, num_steps+1)
        # `pred_len` of shape: (batch_size,)
        pred_token, pred_len = self._get_sentence_token_and_length(logits_1.argmax(dim=2))
        pred_token_mask = get_sequence_mask(pred_len, max_len=pred_token.size(1))
        embeded_pred = self.text_embedding(pred_token)
        # `encoded_pred_outputs` of shape: (batch_size, num_steps+1, hidden_size)
        # `encoded_pred_hidden` of shape: (num_layers, batch_size, hidden_size)
        encoded_pred_outputs, encoded_pred_hidden = self.text_encoder((embeded_pred, pred_len))
        if encoded_pred_outputs.size(1) != embeded_pred.size(1):
            print(pred_len)
            print(pred_token)
            print(encoded_pred_outputs.size(), embeded_pred.size())
            assert 0

        last_predictions = pred_token.new_full((pred_token.size(0),), fill_value=self.start_index)
        step_logits = []
        redecode_hidden = encoded_pred_hidden
        for timestep in range(num_steps):
            if early_stop and (last_predictions == self.end_index).all():
                break
            if self.training and torch.rand(1).item() < teaching_force_rate:
                last_predictions = response[:, timestep]
            # `outputs` of shape (batch_size, num_classes)
            redecode_outputs, redecode_hidden = \
                self._take_step(last_predictions,
                                hidden=redecode_hidden,
                                attn_value=attn_value,
                                attn_mask=attn_mask,
                                pred_value=encoded_pred_outputs,
                                pred_mask=pred_token_mask)
            # shape: (batch_size,)
            last_predictions = torch.argmax(redecode_outputs, dim=-1)
            step_logits.append(redecode_outputs.unsqueeze(1))

        logits_2 = torch.cat(step_logits, dim=1)
        return logits_1, logits_2

    def _take_step(self, inputs, hidden,
                   attn_value, attn_mask,
                   pred_value, pred_mask):
        """
        Redecode decoder step.

        `inputs` of shape: (batch_size,)
        `hidden` of shape: (num_layers, batch_size, hidden_size)
        `attn_value` of shape: (batch_size, num_rows, embedding_size)
        `attn_mask` of shape: (batch_size, num_rows)
        `pred_value` of shape: (batch_size, pred_len, embedding_size)
        `pred_mask` of shape: (batch_size, pred_len)

        """

        # shape: (batch_size, input_size)
        embedded_inputs = self.text_embedding(inputs)
        # shape: (batch_size, num_rows)
        decoder_1_attn_score = self.decoder_1_attn(hidden[-1], attn_value, attn_mask)
        decoder_1_attn_inputs = torch.sum(decoder_1_attn_score.unsqueeze(2) * attn_value, dim=1)

        decoder_2_attn_score = self.decoder_2_attn(hidden[-1], pred_value, pred_mask)
        decoder_2_attn_inputs = torch.sum(decoder_2_attn_score.unsqueeze(2) * pred_value, dim=1)

        # shape: (batch_size, input_size + hidden_size*2)
        decoder_2_inputs = torch.cat([embedded_inputs, decoder_1_attn_inputs, decoder_2_attn_inputs], dim=-1)
        _, next_hidden = self.decoder_2(decoder_2_inputs.unsqueeze(1), hidden)
        # shape: (batch_size, num_classes)
        outputs = self.output_layer(next_hidden[-1])
        return outputs, next_hidden

    def forward_beam_search(self, hidden,
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
        # shape: (batch_size, beam_size, num_steps)
        top_k_pred_token, _ = \
            self.decoder_1.forward_beam_search(hidden=hidden,
                                               attn_value=attn_value,
                                               attn_mask=attn_mask,
                                               num_steps=num_steps,
                                               beam_size=beam_size,
                                               per_node_beam_size=per_node_beam_size,
                                               early_stop=early_stop)
        # `pred_token` of shape: (batch_size, num_steps+1)
        # `pred_len` of shape: (batch_size,)
        pred_token, pred_len = self._get_sentence_token_and_length(top_k_pred_token[:, 0, :])
        pred_token_mask = get_sequence_mask(pred_len, max_len=pred_token.size(1))
        embedded_pred = self.text_embedding(pred_token)
        start_predictions = pred_token.new_full((pred_token.size(0),), fill_value=self.start_index)

        # `encoded_pred_outputs` of shape: (batch_size, num_steps+1, hidden_size)
        # `encoded_pred_hidden` of shape: (num_layers, batch_size, hidden_size)
        encoded_pred_outputs, encoded_pred_hidden = self.text_encoder((embedded_pred, pred_len))

        # `hidden` of shape: (batch_size, num_layers, hidden_size)
        redecode_hidden = encoded_pred_hidden.transpose(0, 1).contiguous()
        state = {'hidden': redecode_hidden,
                 'attn_value': attn_value,
                 'attn_mask': attn_mask,
                 'pred_value': encoded_pred_outputs,
                 'pred_mask': pred_token_mask}
        all_top_k_predictions, _ = \
            beam_search.search(start_predictions, state, self._beam_step, early_stop=early_stop)
        return pred_token, all_top_k_predictions[:, 0, :]

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
        decoder_1_attn_score = self.decoder_attn(hidden[-1], attn_value, attn_mask)
        decoder_1_attn_inputs = torch.sum(decoder_1_attn_score.unsqueeze(2) * attn_value, dim=1)

        pred_value = state['pred_value']
        pred_mask = state['pred_mask']
        decoder_2_attn_score = self.decoder_2_attn(hidden[-1], pred_value, pred_mask)
        decoder_2_attn_inputs = torch.sum(decoder_2_attn_score.unsqueeze(2) * pred_value, dim=1)

        # shape: (group_size, input_size + attn_size)
        decoder_2_inputs = torch.cat([embedded_inputs, decoder_1_attn_inputs, decoder_2_attn_inputs], dim=-1)
        _, next_hidden = self.decoder_2(decoder_2_inputs.unsqueeze(1), hidden)
        state['hidden'] = next_hidden.transpose(0, 1).contiguous()
        # shape: (group_size, num_classes)
        log_prob = F.log_softmax(self.output_layer(next_hidden[-1]), dim=-1)
        return log_prob, state

    def _get_sentence_token_and_length(self, tokens):
        # `tokens` of shape: (batch_size, num_steps)
        batch_size, num_steps = tokens.size()
        tokens = torch.cat([tokens, tokens.new_full((batch_size, 1), fill_value=self.end_index)], dim=1)
        lengths = -((tokens == self.end_index).flip(1)).argmax(dim=1) + num_steps + 1
        return tokens, lengths
