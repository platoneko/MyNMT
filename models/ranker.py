from modules.rnn.rnn_encoder import GRUEncoder
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
            post_tokens, post_len = inputs.post
        else:
            post_tokens = inputs.post
        if isinstance(inputs.response, tuple):
            response_tokens, response_len = inputs.response
        else:
            response_tokens = inputs.response
        embedded_post = self.post_embedding(post_tokens)
        post_mask = inputs.post.ne(self.padding_idx)
        # shape: (batch_size, embedding_size)
        post_vector = masked_max(embedded_post, post_mask.unsqueeze(2), 1)
        batch_size = post_vector.size(0)
        embedded_response = self.response_embedding(response_tokens)
        response_mask = inputs.response.ne(self.padding_idx)
        # shape: (batch_size, embedding_size)
        response_vector = masked_max(embedded_response, response_mask.unsqueeze(2), 1, mask_fill_value=0.0)
        # shape: (batch_size, batch_size)
        score_matrix = post_vector.matmul(response_vector.transpose(0, 1)) / math.sqrt(self.embedding_size)
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
        post_vector = masked_max(embedded_post, post_mask.unsqueeze(2), 1, mask_fill_value=0.0)
        score_matrix = post_vector.matmul(candidate.transpose(0, 1))
        score, indices = score_matrix.topk(topk, 1)
        return score, indices

    def rank_with_sentences(self, post, candidate, topk):
        # `candidate` is candidate sentences token tensor
        embedded_post = self.post_embedding(post)
        post_mask = post.ne(self.padding_idx)
        # shape: (batch_size, embedding_size)
        post_vector = masked_max(embedded_post, post_mask.unsqueeze(2), 1, mask_fill_value=0.0)
        embedded_candidate = self.response_embedding(candidate)
        candidate_mask = candidate.ne(self.padding_idx)
        # shape: (batch_size, embedding_size)
        candidate_vectors = masked_max(embedded_candidate, candidate_mask.unsqueeze(2), 1, mask_fill_value=0.0)
        score_matrix = post_vector.matmul(candidate_vectors.transpose(0, 1))
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
        candidate_vector = masked_max(embedded_candidate, candidate_mask.unsqueeze(2), 1, mask_fill_value=0.0)
        return candidate_vector


class RNNRanker(BaseModel):
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
        self.post_rnn = GRUEncoder(embedding_size, embedding_size//2, 1, bidirectional=True)
        self.response_rnn = GRUEncoder(embedding_size, embedding_size//2, 1, bidirectional=True)

    def forward(self, inputs):
        post_tokens, post_len = inputs.post
        response_tokens, response_len = inputs.response
        embedded_post = self.post_embedding(post_tokens)
        _, post_vector = self.post_rnn((embedded_post, post_len))
        batch_size = post_tokens.size(0)
        embedded_response = self.response_embedding(response_tokens)
        _, response_vector = self.response_rnn((embedded_response, response_len))
        # shape: (batch_size, batch_size)
        score_matrix = post_vector[-1].matmul(response_vector[-1].transpose(0, 1)) / math.sqrt(self.embedding_size)
        truth_score = score_matrix.diagonal(0).unsqueeze(1)
        loss = (score_matrix - truth_score + self.margin).clamp(0.0)
        weight = 1.0 - torch.diag(torch.ones(batch_size, device=embedded_post.device))
        loss = (loss * weight).sum() / (batch_size * (batch_size-1))
        return loss

    def rank_with_tensor(self, post, candidate, topk):
        # `candidate` is candidate vectors tensor
        post_tokens, post_len = post
        embedded_post = self.post_embedding(post_tokens)
        # shape: (batch_size, embedding_size)
        _, post_vector = self.post_rnn((embedded_post, post_len))
        score_matrix = post_vector[-1].matmul(candidate.transpose(0, 1))
        score, indices = score_matrix.topk(topk, 1)
        return score, indices

    def rank_with_sentences(self, post, candidate, topk):
        # `candidate` is candidate sentences token tensor
        post_tokens, post_len = post
        embedded_post = self.post_embedding(post_tokens)
        # shape: (batch_size, embedding_size)
        _, post_vector = self.post_rnn((embedded_post, post_len))
        candidate_tokens, candidate_len = candidate
        embedded_candidate = self.response_embedding(candidate_tokens)
        # shape: (batch_size, embedding_size)
        _, candidate_vectors = self.response_rnn((embedded_candidate, candidate_len))
        score_matrix = post_vector[-1].matmul(candidate_vectors[-1].transpose(0, 1))
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
        batch_size = inputs.post[0].size(0)
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
        candidate_tokens, candidate_len = candidate
        embedded_candidate = self.response_embedding(candidate_tokens)
        # shape: (batch_size, embedding_size)
        _, candidate_vector = self.response_rnn((embedded_candidate, candidate_len))
        return candidate_vector[-1]
