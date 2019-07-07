import torch.nn as nn
from modules import GRUEncoder
from modules import GRUDecoder
from modules import MLPAttention
from utils import *


class Seq2Seq(nn.Module):
    def __init__(
            self,
            embedding,
            embedding_size,
            hidden_size,
            start_index,
            end_index,
            dropout=0.2,
            num_steps=20,
            teaching_force_rate=0.5,
            beam_size=4,
            per_node_beam_size=4
    ):
        self.embedding = embedding
        self.encoder = GRUEncoder(embedding_size, hidden_size // 2, dropout=dropout)
        decoder_attn = MLPAttention(hidden_size, hidden_size, hidden_size)
        decoder_input_size = embedding_size + hidden_size
        num_classes = embedding.weight.size(0)
        self.decoder = GRUDecoder(decoder_input_size,
                                  hidden_size,
                                  num_classes,
                                  start_index,
                                  end_index,
                                  embedding,
                                  attention=decoder_attn,
                                  num_steps=num_steps,
                                  dropout=dropout)
        self.teaching_force_rate = teaching_force_rate
        self.beam_size = beam_size
        self.per_node_beam_size = per_node_beam_size

    def forward(self, post, response=None, training=False):
        if training:
            assert response is not None
        if response is not None:
            response_token, response_len = response

        post_token, post_len = post
        embedded_post = self.embedding(post_token)
        encoder_outputs, encoder_hidden = self.encoder((embedded_post, post_len))
        encoder_outputs_mask = get_sequence_mask(post_len)
        if training:
            logits, predictions = self.decoder(
                encoder_hidden[0],
                target=response,
                attn_value=encoder_outputs,
                attn_mask=encoder_outputs_mask,
                teaching_force_rate=self.teaching_force_rate
            )
            target = response_token[:, 1:].continuous()
            target_mask = get_sequence_mask(response_len - 1)
            loss = sequence_cross_entropy_with_logits(logits, target, target_mask)
            return loss, predictions
        else:
            # eval
            if response is not None:
                logits, predictions = self.decoder(
                    encoder_hidden[0],
                    target=response,
                    attn_value=encoder_outputs,
                    attn_mask=encoder_outputs_mask,
                    teaching_force_rate=self.teaching_force_rate
                )
                target = response_token[:, 1:].continuous()
                target_mask = get_sequence_mask(response_len - 1)
                loss = sequence_cross_entropy_with_logits(logits, target, target_mask)
                if self.beam_size > 1:
                    all_top_k_predictions, log_probabilities = self.decoder.forward_beam_search(
                        encoder_hidden[0],
                        encoder_outputs,
                        encoder_outputs_mask,
                        beam_size=self.beam_size,
                        per_node_beam_size=self.per_node_beam_size
                    )
                    predictions = all_top_k_predictions[:, 0, :]
                return loss, predictions
            # test
            else:
                if self.beam_size <= 1:
                    logits, predictions = self.decoder(
                        encoder_hidden[0],
                        target=response,
                        attn_value=encoder_outputs,
                        attn_mask=encoder_outputs_mask,
                        teaching_force_rate=self.teaching_force_rate
                    )
                else:
                    all_top_k_predictions, log_probabilities = self.decoder.forward_beam_search(
                        encoder_hidden[0],
                        encoder_outputs,
                        encoder_outputs_mask,
                        beam_size=self.beam_size,
                        per_node_beam_size=self.per_node_beam_size
                    )
                    predictions = all_top_k_predictions[:, 0, :]
                return predictions
