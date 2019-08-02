from models.base_model import BaseModel
from modules.rnn import GRUEncoder
from modules.utils import *
from utils.pack import Pack
from overrides import overrides

import torch
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn


class ConvClassifier(BaseModel):
    """
    Supervised Classify Models
    """
    def __init__(self,
                 embedding_size,
                 response_embedding,
                 num_classes,
                 padding_idx,
                 ):
        super().__init__()
        self.embedding_size = embedding_size
        self.response_embedding = response_embedding
        self.padding_idx = padding_idx
        self.num_classes = num_classes

        self.conv_layer = nn.Sequential(
            nn.Conv1d(embedding_size, 500, 3),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

        # self.rnn = GRUEncoder(embedding_size, embedding_size//2, bidirectional=True)
        self.output_layer = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(embedding_size, num_classes),
            nn.Softmax(dim=-1))
        self.nll_loss = nn.NLLLoss()

    def forward(self, inputs):
        response, lengths = inputs.response
        embedded_response = self.response_embedding(response)
        response_mask = response.ne(self.padding_idx)
        masked_vector = embedded_response.masked_fill((1 - response_mask.unsqueeze(2)).byte(), 0.0)
        # response_vectors = masked_max(embedded_response, response_mask.unsqueeze(2), 1)
        # shape: (batch_size, embedding_size)
        response_vectors, _ = self.conv_layer(masked_vector.transpose(1, 2)).max(2)
        # _, response_vectors = self.rnn((embedded_response, lengths))
        # shape: (batch_size, batch_size)
        probabilities = self.output_layer(response_vectors)
        outputs = Pack(probabilities=probabilities, vectors=response_vectors)
        return outputs

    @overrides
    def collect_metrics(self, probabilities, speaker):
        """
        collect_metrics
        """
        num_samples = probabilities.size(0)
        metrics = Pack(num_samples=num_samples)
        loss = self.nll_loss(torch.log(probabilities), speaker)
        acc = (probabilities.argmax(1) == speaker).sum().float() / num_samples
        metrics.add(loss=loss, acc=acc)
        return metrics

    @overrides
    def iterate(self, inputs, optimizer=None, grad_clip=None):
        """
        iterate
        """
        outputs = self.forward(inputs)
        metrics = self.collect_metrics(outputs.probabilities, inputs.speaker)

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

