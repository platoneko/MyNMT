#!/usr/bin/env python
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: source/models/seq2seq.py
"""

import math
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from models.base_model import BaseModel
from modules.rnn.rnn_encoder import GRUEncoder
from modules.rnn.baidu_decoder import RNNDecoder
from utils.metrics import accuracy
from utils.pack import Pack


class BaiduSeq2Seq(BaseModel):
    """
    Seq2Seq
    """
    def __init__(self,
                 post_embedding,
                 response_embedding,
                 embed_size,
                 hidden_size,
                 padding_idx,
                 end_idx,
                 num_layers=1,
                 bidirectional=True,
                 attn_mode="mlp",
                 with_bridge=False,
                 dropout=0.0):
        super().__init__()

        self.post_embedding = post_embedding
        self.response_embedding = response_embedding

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx
        self.end_idx = end_idx
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.attn_mode = attn_mode
        self.attn_hidden_size = hidden_size
        self.with_bridge = with_bridge
        self.dropout = dropout

        self.encoder = GRUEncoder(input_size=self.embed_size,
                                  hidden_size=self.hidden_size//2,
                                  num_layers=self.num_layers,
                                  bidirectional=self.bidirectional,
                                  dropout=self.dropout)

        if self.with_bridge:
            self.bridge = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.Tanh(),
            )
        self.num_classes = response_embedding.weight.size(0)
        self.decoder = RNNDecoder(input_size=self.embed_size,
                                  hidden_size=self.hidden_size,
                                  output_size=self.num_classes,
                                  embedder=response_embedding,
                                  num_layers=self.num_layers,
                                  attn_mode=self.attn_mode,
                                  attn_hidden_size=self.attn_hidden_size,
                                  memory_size=self.hidden_size,
                                  feature_size=None,
                                  dropout=self.dropout)

        # Loss Definition
        self.nll_loss = nn.NLLLoss(ignore_index=padding_idx)

    def encode(self, inputs):
        """
        encode
        """
        outputs = Pack()
        enc_inputs, lengths = inputs
        embed_enc = self.post_embedding(enc_inputs)
        enc_outputs, enc_hidden = self.encoder((embed_enc, lengths))

        if self.with_bridge:
            enc_hidden = self.bridge(enc_hidden)

        dec_init_state = self.decoder.initialize_state(
            hidden=enc_hidden,
            attn_memory=enc_outputs if self.attn_mode else None,
            memory_lengths=lengths if self.attn_mode else None)
        return outputs, dec_init_state

    def decode(self, input, state):
        """
        decode
        """
        log_prob, state, output = self.decoder.decode(input, state)
        return log_prob, state, output

    def forward(self, enc_inputs, dec_inputs):
        """
        forward
        """
        outputs, dec_init_state = self.encode(enc_inputs)
        log_probs, _ = self.decoder(dec_inputs, dec_init_state)
        outputs.add(logits=log_probs)
        return outputs

    def collect_metrics(self, outputs, target):
        """
        collect_metrics
        """
        num_samples = target.size(0)
        metrics = Pack(num_samples=num_samples)
        loss = 0

        logits = outputs.logits
        nll = self.nll_loss(logits.reshape(-1, self.num_classes), target.reshape(-1))
        predictions = logits.argmax(dim=2)
        acc = accuracy(predictions, target, padding_idx=self.padding_idx, end_idx=self.end_idx)
        metrics.add(nll=nll, acc=acc)
        metrics.add(ppx=math.exp(nll.item()))
        loss += nll

        metrics.add(loss=loss)
        return metrics

    def iterate(self, inputs, optimizer=None, grad_clip=None, epoch=-1):
        """
        iterate
        """
        enc_inputs = inputs.post
        dec_inputs = inputs.response[0][:, :-1], inputs.response[1] - 1
        outputs = self.forward(enc_inputs, dec_inputs)
        target = inputs.response[0][:, 1:]
        metrics = self.collect_metrics(outputs, target)

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
