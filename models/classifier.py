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
                 text_embedding,
                 kernel_size,
                 num_classes,
                 padding_idx,
                 ):
        super().__init__()
        self.embedding_size = embedding_size
        self.text_embedding = text_embedding
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

    def forward(self, input, mask=None):
        embedded_input = self.text_embedding(input)
        if mask is None:
            mask = input.ne(self.padding_idx)
        masked_vector = embedded_input.masked_fill((1 - mask.unsqueeze(2)).byte(), 0.0)
        internal = self.conv_layer(masked_vector.transpose(1, 2))
        response_vector = F.max_pool1d(internal, internal.size(2)).squeeze(2)
        # shape: (batch_size, batch_size)
        probability = self.output_layer(response_vector)
        outputs = Pack(probability=probability, vector=response_vector)
        return outputs

    @overrides
    def collect_metrics(self, probability, speaker):
        """
        collect_metrics
        """
        num_samples = probability.size(0)
        metrics = Pack(num_samples=num_samples)
        loss = self.focal_loss(torch.log(probability), speaker)
        acc = (probability.argmax(1) == speaker).sum().float() / num_samples
        metrics.add(loss=loss, acc=acc)
        return metrics

    @overrides
    def iterate(self, inputs, optimizer=None, grad_clip=None):
        # inputs should contain `origin` and `speaker`

        if isinstance(inputs.origin, tuple):
            input, _ = inputs.origin
        else:
            input = inputs.origin
        outputs = self.forward(input)
        metrics = self.collect_metrics(outputs.probability, inputs.speaker)

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


class DeltaConvClassifier(BaseModel):
    """
    Supervised Classify Models
    """
    def __init__(self,
                 embedding_size,
                 text_embedding,
                 kernel_size,
                 num_classes,
                 padding_idx,
                 ):
        super().__init__()
        self.embedding_size = embedding_size
        self.text_embedding = text_embedding
        self.padding_idx = padding_idx
        self.num_classes = num_classes

        self.conv_layer = nn.Sequential(
            nn.Conv1d(embedding_size, embedding_size, kernel_size, padding=kernel_size-1),
            nn.ReLU(),
        )

        self.output_layer = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(embedding_size, num_classes, bias=False),
            nn.Softmax(dim=-1))
        self.focal_loss = FocalLoss(reduce=True)

    def forward(self, origin, destylized, origin_mask=None, destylized_mask=None):
        embedded_origin = self.text_embedding(origin)
        if origin_mask is None:
            origin_mask = origin.ne(self.padding_idx)
        masked_origin = embedded_origin.masked_fill((1 - origin_mask.unsqueeze(2)).byte(), 0.0)
        internal = self.conv_layer(masked_origin.transpose(1, 2))
        response_vector = F.max_pool1d(internal, internal.size(2)).squeeze(2)

        embedded_destylized = self.text_embedding(destylized)
        if destylized_mask is None:
            destylized_mask = destylized.ne(self.padding_idx)
        masked_destylized = embedded_destylized.masked_fill((1 - destylized_mask.unsqueeze(2)).byte(), 0.0)
        internal = self.conv_layer(masked_destylized.transpose(1, 2))
        destylized_vector = F.max_pool1d(internal, internal.size(2)).squeeze(2)

        style_vector = response_vector - destylized_vector
        probability = self.output_layer(style_vector)
        outputs = Pack(probability=probability, vector=style_vector)
        return outputs

    @overrides
    def collect_metrics(self, probability, speaker):
        """
        collect_metrics
        """
        num_samples = probability.size(0)
        metrics = Pack(num_samples=num_samples)
        loss = self.focal_loss(torch.log(probability), speaker)
        acc = (probability.argmax(1) == speaker).sum().float() / num_samples
        metrics.add(loss=loss, acc=acc)
        return metrics

    @overrides
    def iterate(self, inputs, optimizer=None, grad_clip=None):
        # inputs should contain `origin`, `destylized` and `speaker`

        if isinstance(inputs.origin, tuple):
            origin, _ = inputs.origin
        else:
            origin = inputs.origin
        if isinstance(inputs.destylized, tuple):
            destylized, _ = inputs.destylized
        else:
            destylized = inputs.destylized
        outputs = self.forward(origin, destylized)
        metrics = self.collect_metrics(outputs.probability, inputs.speaker)

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
