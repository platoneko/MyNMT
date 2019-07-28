from modules.rnn import GRUEncoder
from models.base_model import BaseModel
from modules.utils import *


class IRRanker(BaseModel):
    """
    Supervised Embedding Models
    """
    def __init__(self,
                 post_embedding,
                 response_embedding,
                 padding_idx
                 ):
        super().__init__()
        self.post_embedding = post_embedding
        self.response_embedding = response_embedding
        self.padding_idx = padding_idx

    def forward(self, inputs):
        embedded_post = self.post_embedding(inputs.post)
        post_mask = inputs.post.ne(self.padding_idx)
        # shape: (batch_size, embedding_size)
        post_vectors = masked_sum(embedded_post, post_mask, 2)

        embedded_response = self.post_embedding(inputs.response)
        response_mask = inputs.response.ne(self.padding_idx)
        # shape: (batch_size, embedding_size)
        response_vectors = masked_sum(embedded_response, response_mask, 2)

        