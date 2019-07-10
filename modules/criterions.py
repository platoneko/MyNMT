import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class SequenceNLLLoss(_Loss):
    """
    NLLLoss for sequence, usually ignores padding index
    """
    def __init__(self, weight=None, ignore_index=-100, reduction='mean'):
        super().__init__()
        assert reduction in ['none', 'sum', 'mean']
        self.register_buffer('weight', weight)
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, inputs, target):
        """
        inputs: (batch_size, max_len, num_classes)
        target: (batch_size, max_len)
        """
        nll = F.nll_loss(
            input=inputs.reshape(-1, inputs.size(-1)),
            target=target.reshape(-1),
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction
        )
        return nll


class SequenceCrossEntropy(_Loss):
    """
    Cross entropy for sequence, usually ignores padding index
    """
    def __init__(self, weight=None, ignore_index=-100, reduction='mean'):
        super().__init__()
        assert reduction in ['none', 'sum', 'mean']
        self.register_buffer('weight', weight)
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, inputs, target):
        """
        inputs: (batch_size, max_len, num_classes)
        target: (batch_size, max_len)
        """
        cross_entropy = F.cross_entropy(
            input=inputs.reshape(-1, inputs.size(-1)),
            target=target.reshape(-1),
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction
        )
        return cross_entropy
