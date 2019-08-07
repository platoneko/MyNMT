from models.base_model import BaseModel
from modules.criterions import FocalLoss
from utils.pack import Pack
from overrides import overrides

import torch
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn
import torch.nn.functional as F


class ConvClassifier(BaseModel):
    """
    Supervised Classify Models
    """
    def __init__(self,
                 embedding_size,
                 response_embedding,
                 kernel_size,
                 num_classes,
                 padding_idx,
                 ):
        super().__init__()
        self.embedding_size = embedding_size
        self.response_embedding = response_embedding
        self.padding_idx = padding_idx
        self.num_classes = num_classes

        self.conv_layer = nn.Sequential(
            nn.Conv1d(embedding_size, embedding_size, kernel_size),
            nn.ReLU(),
        )

        self.output_layer = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(embedding_size, num_classes),
            nn.Softmax(dim=-1))
        self.focal_loss = FocalLoss(reduce=True)

    def forward(self, inputs, mask=None):
        embedded_inputs = self.response_embedding(inputs)
        if mask is None:
            mask = inputs.ne(self.padding_idx)
        masked_vector = embedded_inputs.masked_fill((1 - mask.unsqueeze(2)).byte(), 0.0)
        internal = self.conv_layer(masked_vector.transpose(1, 2))
        response_vectors = F.max_pool1d(internal, internal.size(2)).squeeze(2)
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
        loss = self.focal_loss(torch.log(probabilities), speaker)
        acc = (probabilities.argmax(1) == speaker).sum().float() / num_samples
        metrics.add(loss=loss, acc=acc)
        return metrics

    @overrides
    def iterate(self, inputs, optimizer=None, grad_clip=None):
        """
        iterate
        """
        if isinstance(inputs.response, tuple):
            response, _ = inputs.response
        else:
            response = inputs.response
        outputs = self.forward(response)
        metrics = self.collect_metrics(outputs.probabilities, inputs.speaker)

        loss = metrics.loss
        if torch.isnan(loss):
            print(outputs.probabilities, outputs.vectors, self.conv_layer[0].weight.data)
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
