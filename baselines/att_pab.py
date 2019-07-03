import torch.nn as nn
from modules import GRUEncoder
from modules import MLPAttention


class AttPAB(nn.Module):
    def __init__(
            self,
            text_embedding,
            gender_embedding,
            loc_embedding,
            embedding_size,
            hidden_size,
            dropout=0.2
    ):
        self.text_embedding = text_embedding
        self.gender_embedding = gender_embedding
        self.loc_embedding = loc_embedding
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.encoder = GRUEncoder(embedding_size, hidden_size // 2, dropout=dropout)
        self.profile_attn = MLPAttention(hidden_size, embedding_size, hidden_size)
        self.tag_attn = MLPAttention(hidden_size, embedding_size, hidden_size)
        self.decoder_attn = MLPAttention(hidden_size, hidden_size, hidden_size)

    def forward(self, post, loc, gender, tag, response=None, training=True):
        if training:
            assert response is not None
        post_token, post_len = post
        embedded_post = self.text_embedding(post_token)
        # shape of `encoder_outputs`: (batch, len, hidden_size);
        # shape of `encoder_hidden`: (num_layers, batch, len, hidden_size)
        encoder_outputs, encoder_hidden = self.encoder((embedded_post, post_len))


class AttPABDecoder(nn.Module):
