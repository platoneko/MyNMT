import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.utils import *
from modules.rnn.rnn_encoder import GRUEncoder


class MemoryNetwork(nn.Module):
    """
    Memory Network in `End-To-End Memory Networks` with adjacent weight tying.
    """

    def __init__(self, num_hops, vocab_size, embedding_size, padding_index):
        super().__init__()
        self.num_hops = num_hops
        self.memory_embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size, embedding_size) for _ in range(num_hops+1)])
        self.padding_index = padding_index

    def forward(self, query, memory):
        # `query` of shape (batch_size, embedding_size)
        # `memory` of shape (batch_size, k, length)
        q = query
        memory_mask = memory.ne(self.padding_index)
        # shape: (batch_size, k, embedding_size)
        C = masked_max(self.memory_embeddings[0](memory), memory_mask.unsqueeze(-1), dim=2, mask_fill_value=0.0)
        for i in range(self.num_hops):
            # shape: (batch_size, k, 1)
            p = F.softmax(C.matmul(q.unsqueeze(-1)), dim=1)
            # shape: (batch_size, k, embedding_size)
            C_ = masked_max(self.memory_embeddings[i + 1](memory), memory_mask.unsqueeze(-1), dim=2, mask_fill_value=0.0)
            o = p.transpose(1, 2).matmul(C_).squeeze(1)
            q = q + o
            C = C_
        return q
