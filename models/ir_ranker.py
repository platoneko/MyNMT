from models.base_model import BaseModel
from modules.utils import *


class IRRanker(BaseModel):
    """
    Supervised Embedding Models
    """
    def __init__(self,
                 post_embedding,
                 response_embedding,
                 padding_idx,
                 margin=1.0
                 ):
        super().__init__()
        self.post_embedding = post_embedding
        self.response_embedding = response_embedding
        self.padding_idx = padding_idx
        self.margin = margin

    def forward(self, inputs):
        embedded_post = self.post_embedding(inputs.post)
        post_mask = inputs.post.ne(self.padding_idx)
        # shape: (batch_size, embedding_size)
        post_vectors = masked_sum(embedded_post, post_mask.unsqueeze(2), 2)
        batch_size = post_vectors.size(0)
        embedded_response = self.response_embedding(inputs.response)
        response_mask = inputs.response.ne(self.padding_idx)
        # shape: (batch_size, embedding_size)
        response_vectors = masked_sum(embedded_response, response_mask.unsqueeze(2), 2)
        # shape: (batch_size, batch_size)
        score_matrix = post_vectors.matmul(response_vectors.transpose(0, 1))
        truth_score = score_matrix.diagonal(score_matrix, 0).unsqueeze(1)
        loss = (score_matrix - truth_score + self.margin).clamp(0.0)
        weight = 1.0 - torch.diag(torch.ones(batch_size))
        loss = loss * weight / batch_size
        return loss

    def rank_with_tensor(self, post, candidate, topk):
        embedded_post = self.post_embedding(post)
        post_mask = post.ne(self.padding_idx)
        # shape: (batch_size, embedding_size)
        post_vectors = masked_sum(embedded_post, post_mask.unsqueeze(2), 2)
        score_matrix = post_vectors.matmul(candidate.transpose(0, 1))
        score, indices = score_matrix.topk(topk, 1)
        return score, indices

    def rank_with_sentences(self, post, candidate, topk):
        embedded_post = self.post_embedding(post)
        post_mask = post.ne(self.padding_idx)
        # shape: (batch_size, embedding_size)
        post_vectors = masked_sum(embedded_post, post_mask.unsqueeze(2), 2)
        embedded_candidate = self.response_embedding(candidate)
        candidate_mask = candidate.ne(self.padding_idx)
        # shape: (batch_size, embedding_size)
        candidate_vectors = masked_sum(embedded_candidate, candidate_mask.unsqueeze(2), 2)
        score_matrix = post_vectors.matmul(candidate_vectors.transpose(0, 1))
        score, indices = score_matrix.topk(topk, 1)
        return score, indices
