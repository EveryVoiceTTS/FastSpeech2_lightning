"""Conformer that keeps padding from leaking into the real positions.

torchaudio's Conformer masks padded positions in self-attention, but not in the
convolution module: its depthwise convolution reads padded positions, which are
not zero after the preceding layers, so each item's output depends on how much
padding the rest of its batch added. Stacked layers spread that leak over the
whole sequence.

These subclasses zero the padded positions right before the depthwise
convolution. They keep torchaudio's module structure and attribute names, so
state_dict keys are unchanged and existing checkpoints load as before. Without
padding (e.g., batch size 1), the output is identical to torchaudio's.
"""

from typing import Optional

import torch
from torchaudio.models import Conformer
from torchaudio.models.conformer import ConformerLayer


class MaskedConformerLayer(ConformerLayer):
    """ConformerLayer that zeroes padded positions before its depthwise conv."""

    def _apply_masked_convolution(
        self, input: torch.Tensor, key_padding_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Same as ConformerLayer._apply_convolution, but with padding masked.

        Args:
            input: shape `(T, B, D)`
            key_padding_mask: shape `(B, T)`, True at padded positions
        """
        conv_module = self.conv_module
        x = conv_module.layer_norm(input.transpose(0, 1))  # (B, T, D)
        x = x.transpose(1, 2)  # (B, D, T)
        pointwise_conv, glu, *rest = conv_module.sequential
        x = glu(pointwise_conv(x))
        # Mask after the pointwise conv and GLU, since the pointwise conv has a
        # bias: this is the first layer that mixes time steps.
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(1), 0.0)
        for layer in rest:
            x = layer(x)
        return input + x.transpose(1, 2).transpose(0, 1)

    def forward(
        self, input: torch.Tensor, key_padding_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Same as ConformerLayer.forward, but with the masked convolution.

        Args:
            input: shape `(T, B, D)`
            key_padding_mask: shape `(B, T)`, True at padded positions

        Returns:
            output, with shape `(T, B, D)`
        """
        residual = input
        x = self.ffn1(input)
        x = x * 0.5 + residual

        if self.convolution_first:
            x = self._apply_masked_convolution(x, key_padding_mask)

        residual = x
        x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = self.self_attn_dropout(x)
        x = x + residual

        if not self.convolution_first:
            x = self._apply_masked_convolution(x, key_padding_mask)

        residual = x
        x = self.ffn2(x)
        x = x * 0.5 + residual

        x = self.final_layer_norm(x)
        return x


class MaskedConformer(Conformer):
    """torchaudio Conformer built from MaskedConformerLayer layers."""

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_layers: int,
        depthwise_conv_kernel_size: int,
        dropout: float = 0.0,
        use_group_norm: bool = False,
        convolution_first: bool = False,
    ):
        # Skip Conformer.__init__, which only builds conformer_layers: building
        # them twice would consume extra random numbers and change the initial
        # weights of seeded models compared with torchaudio's Conformer.
        torch.nn.Module.__init__(self)
        self.conformer_layers = torch.nn.ModuleList(
            [
                MaskedConformerLayer(
                    input_dim,
                    ffn_dim,
                    num_heads,
                    depthwise_conv_kernel_size,
                    dropout=dropout,
                    use_group_norm=use_group_norm,
                    convolution_first=convolution_first,
                )
                for _ in range(num_layers)
            ]
        )
