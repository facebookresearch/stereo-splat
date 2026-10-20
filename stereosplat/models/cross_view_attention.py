# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Cross-view attention modules for feature-level multi-view fusion.

Provides a lightweight transformer that operates across views at each spatial
position.  Sequence length = N (number of views, typically 2-6), so the
attention matrix is tiny and the compute overhead is minimal.

All output projections and FFN final layers are zero-initialised so the model
starts as an identity (baseline-equivalent).
"""

from __future__ import annotations

import torch
from torch import nn


class CrossViewTransformerBlock(nn.Module):
    """Pre-norm transformer block operating across N views per spatial position."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_ratio: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        ffn_hidden = embed_dim * ffn_ratio
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_hidden),
            nn.GELU(),
            nn.Linear(ffn_hidden, embed_dim),
        )

        # Zero-init output projections -> block starts as identity.
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)
        ffn_out = self.ffn[2]
        assert isinstance(ffn_out, nn.Linear)
        nn.init.zeros_(ffn_out.weight)
        nn.init.zeros_(ffn_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B*H*W, N, C]  (N = number of views)

        Returns:
            [B*H*W, N, C]
        """
        h = self.norm1(x)
        h = self.attn(h, h, h, need_weights=False)[0]
        x = x + h

        h = self.norm2(x)
        h = self.ffn(h)
        x = x + h
        return x


class CrossViewTransformer(nn.Module):
    """Stack of cross-view transformer blocks with reshape bookkeeping.

    Reshapes ``[B*N, C, H, W]`` feature maps into ``[B*H*W, N, C]`` tokens,
    applies N stacked transformer blocks, and reshapes back.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        num_layers: int = 4,
        ffn_ratio: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                CrossViewTransformerBlock(embed_dim, num_heads, ffn_ratio, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor, B: int, N: int) -> torch.Tensor:
        """
        Args:
            x: [B*N, C, H, W]  feature maps (all views concatenated).
            B: batch size.
            N: number of views.

        Returns:
            [B*N, C, H, W]  fused feature maps.
        """
        C, H, W = x.shape[1], x.shape[2], x.shape[3]

        # [B*N, C, H, W] -> [B, N, C, H*W] -> [B, H*W, N, C] -> [B*H*W, N, C]
        x = x.view(B, N, C, H * W)
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, H*W, N, C]
        x = x.view(B * H * W, N, C)

        for layer in self.layers:
            x = layer(x)

        # [B*H*W, N, C] -> [B, H*W, N, C] -> [B, N, C, H*W] -> [B*N, C, H, W]
        x = x.view(B, H * W, N, C)
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, N, C, H*W]
        x = x.view(B * N, C, H, W)
        return x
