import torch.nn as nn
from modules import GRUEncoder
from modules import GRUDecoder
from modules import MLPAttention


class Seq2Seq(nn.Module):
    def __init__(
            self,
            embedding,
            embedding_size,
            hidden_size,
            start_index,
            dropout=0.2,
            num_steps=20,
            teaching_force_rate=0.5
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
                                  embedding,
                                  attention=decoder_attn,
                                  num_steps=num_steps,
                                  dropout=dropout)

    def forward(self, post, response=None, training=False):
        post_token, post_len = post
        embedded_post = self.embedding(post_token)
        encoder_outputs, encoder_hidden = self.encoder((embedded_post, post_len))

