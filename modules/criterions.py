import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class SequenceNLLLoss(_Loss):
    """
    NLLLoss for sequence, average/sum the loss across the batches
    """
    def __init__(self, weight=None, padding_idx=None, reduction='mean'):
        super().__init__()
        assert reduction in ['none', 'sum', 'mean']
        self.register_buffer('weight', weight)
        self.padding_idx = padding_idx
        self.reduction = reduction

    def forward(self, input, target, reduction=True):
        """
        input: (batch_size, max_len, vocab_size)
        target: (batch_size, max_len)
        """
        if self.weight is None and self.padding_idx is not None:
            self.weight = input.new_ones(input.size(-1))
        batch_size = input.size(0)
        nll = F.nll_loss(
            input=input.reshape(-1, input.size(-1)),
            target=target.reshape(-1),
            weight=self.weight,
            reduction='none'
        )
        nll = nll.view(batch_size, -1).sum(dim=1)

        if reduction:
            if self.reduction == 'mean':
                nll = nll.mean()
            elif self.reduction == 'sum':
                nll = nll.sum()

        return nll


class SequenceCrossEntropy(_Loss):
    """
    Cross entropy for sequence, average/sum the loss across the batches
    """
    def __init__(self, weight=None, padding_idx=None, reduction='mean'):
        super().__init__()
        assert reduction in ['none', 'sum', 'mean']
        self.register_buffer('weight', weight)
        self.padding_idx = padding_idx
        self.reduction = reduction

    def forward(self, input, target, reduction=True):
        """
        input: (batch_size, max_len, vocab_size)
        target: (batch_size, max_len)
        """
        if self.weight is None and self.padding_idx is not None:
            self.weight = input.new_ones(input.size(-1))
        batch_size = input.size(0)
        cross_entropy = F.cross_entropy(
            input=input.reshape(-1, input.size(-1)),
            target=target.reshape(-1),
            weight=self.weight,
            reduction='none'
        )
        cross_entropy = cross_entropy.view(batch_size, -1).sum(dim=1)

        if reduction:
            if self.reduction == 'mean':
                cross_entropy = cross_entropy.mean()
            elif self.reduction == 'sum':
                cross_entropy = cross_entropy.sum()

        return cross_entropy
