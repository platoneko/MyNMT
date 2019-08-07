from models.base_model import BaseModel
from utils.pack import Pack
from modules.utils import *
from overrides import overrides
from torch.nn.utils import clip_grad_norm_
import math


class EmbeddingRanker(BaseModel):
    """
    Supervised Embedding Ranking Models
    """
    def __init__(self,
                 embedding_size,
                 post_embedding,
                 response_embedding,
                 padding_idx,
                 margin=1.0
                 ):
        super().__init__()
        self.embedding_size = embedding_size
        self.post_embedding = post_embedding
        self.response_embedding = response_embedding
        self.padding_idx = padding_idx
        self.margin = margin

    def forward(self, inputs):
        if isinstance(inputs.post, tuple):
            post, post_len = inputs.post
        else:
            post = inputs.post
        if isinstance(inputs.response, tuple):
            response, response_len = inputs.response
        else:
            response = inputs.response
        embedded_post = self.post_embedding(post)
        post_mask = inputs.post.ne(self.padding_idx)
        # shape: (batch_size, embedding_size)
        post_vectors = masked_sum(embedded_post, post_mask.unsqueeze(2), 1)
        batch_size = post_vectors.size(0)
        embedded_response = self.response_embedding(response)
        response_mask = inputs.response.ne(self.padding_idx)
        # shape: (batch_size, embedding_size)
        response_vectors = masked_sum(embedded_response, response_mask.unsqueeze(2), 1)
        # shape: (batch_size, batch_size)
        score_matrix = post_vectors.matmul(response_vectors.transpose(0, 1)) / math.sqrt(self.embedding_size)
        truth_score = score_matrix.diagonal(0).unsqueeze(1)
        loss = (score_matrix - truth_score + self.margin).clamp(0.0)
        weight = 1.0 - torch.diag(torch.ones(batch_size, device=embedded_post.device))
        loss = (loss * weight).sum() / (batch_size * (batch_size-1))
        return loss

    def rank_with_tensor(self, post, candidate, topk):
        # `candidate` is candidate vectors tensor
        embedded_post = self.post_embedding(post)
        post_mask = post.ne(self.padding_idx)
        # shape: (batch_size, embedding_size)
        post_vectors = masked_sum(embedded_post, post_mask.unsqueeze(2), 1)
        score_matrix = post_vectors.matmul(candidate.transpose(0, 1))
        score, indices = score_matrix.topk(topk, 1)
        return score, indices

    def rank_with_sentences(self, post, candidate, topk):
        # `candidate` is candidate sentences token tensor
        embedded_post = self.post_embedding(post)
        post_mask = post.ne(self.padding_idx)
        # shape: (batch_size, embedding_size)
        post_vectors = masked_sum(embedded_post, post_mask.unsqueeze(2), 1)
        embedded_candidate = self.response_embedding(candidate)
        candidate_mask = candidate.ne(self.padding_idx)
        # shape: (batch_size, embedding_size)
        candidate_vectors = masked_sum(embedded_candidate, candidate_mask.unsqueeze(2), 1)
        score_matrix = post_vectors.matmul(candidate_vectors.transpose(0, 1))
        score, indices = score_matrix.topk(topk, 1)
        return score, indices

    @overrides
    def collect_metrics(self, loss, num_samples):
        """
        collect_metrics
        """
        metrics = Pack(num_samples=num_samples)
        metrics.add(loss=loss)
        return metrics

    @overrides
    def iterate(self, inputs, optimizer=None, grad_clip=None):
        """
        iterate
        """
        loss = self.forward(inputs)
        batch_size = inputs.post.size(0)
        metrics = self.collect_metrics(loss, batch_size)

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

    def get_candidate_vector(self, candidate):
        # `candidate` is candidate sentences token tensor
        embedded_candidate = self.response_embedding(candidate)
        candidate_mask = candidate.ne(self.padding_idx)
        # shape: (num_samples, embedding_size)
        candidate_vectors = masked_sum(embedded_candidate, candidate_mask.unsqueeze(2), 1)
        return candidate_vectors
