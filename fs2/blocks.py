from typing import Optional

import torch
from torch import nn


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(
                in_channels,
                in_channels,
                kernel_size,
                padding=(kernel_size - 1) // 2,
                groups=in_channels,
            ),
            nn.Conv1d(in_channels, out_channels, 1),
        )

    def forward(self, x):
        return self.model(x)


class MaskedBatchNorm1d(nn.BatchNorm1d):
    """BatchNorm1d whose batch statistics only cover valid (non-padded) positions.

    Without a mask, or when using running statistics (i.e., in eval mode), this
    is BatchNorm1d. The parameters and buffers are BatchNorm1d's, so the
    state_dict is the same.
    """

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: shape `(B, C, T)`
            mask: optional, shape `(B, T)`, true or 1 at valid positions.
        """
        use_batch_stats = self.training or not self.track_running_stats
        if mask is None or not use_batch_stats:
            return super().forward(x)

        weights = mask.to(x.dtype).unsqueeze(1)  # (B, 1, T)
        n = weights.sum()
        mean = (x * weights).sum(dim=(0, 2)) / n
        centered = x - mean[None, :, None]
        var = (centered.pow(2) * weights).sum(dim=(0, 2)) / n

        if self.training and self.track_running_stats:
            assert self.num_batches_tracked is not None
            self.num_batches_tracked.add_(1)
            factor = (
                1.0 / float(self.num_batches_tracked)
                if self.momentum is None
                else self.momentum
            )
            with torch.no_grad():
                unbiased_var = var * n / (n - 1).clamp(min=1)
                self.running_mean.lerp_(mean, factor)  # type: ignore[union-attr]
                self.running_var.lerp_(unbiased_var, factor)  # type: ignore[union-attr]

        x = centered / torch.sqrt(var[None, :, None] + self.eps)
        if self.affine:
            x = x * self.weight[None, :, None] + self.bias[None, :, None]
        return x


class LayerNorm(nn.LayerNorm):
    """Layer normalization module.
    :param int nout: output dim size
    :param int dim: dimension to be normalized
    """

    def __init__(self, nout, dim=-1):
        """Construct an LayerNorm object."""
        super(LayerNorm, self).__init__(nout, eps=1e-12)
        self.dim = dim

    def forward(self, x):
        """Apply layer normalization.
        :param torch.Tensor x: input tensor
        :return: layer normalized tensor
        :rtype torch.Tensor
        """
        if self.dim == -1:
            return super(LayerNorm, self).forward(x)
        return super(LayerNorm, self).forward(x.transpose(1, -1)).transpose(1, -1)


class ConvNorm(nn.Module):
    """1D Convolution"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        w_init_gain="linear",
        transpose=False,
    ):
        super(ConvNorm, self).__init__()

        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        nn.init.xavier_uniform_(
            self.conv.weight, gain=nn.init.calculate_gain(w_init_gain)
        )
        self.transpose = transpose

    def forward(self, x):
        if self.transpose:
            x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        if self.transpose:
            x = x.contiguous().transpose(1, 2)

        return x
