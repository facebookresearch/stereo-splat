# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""StereoSplat: metric-scale Gaussian splatting grounded on stereo depth.

Pipeline for a window of ``P`` rectified stereo pairs (``N = 2P`` views):

1. A frozen FoundationStereo backbone predicts left and right disparity for
   every pair (the right one by flipping the pair horizontally) and exposes its
   multi-scale context features. Disparity is converted to metric depth with
   the stereo baseline and focal length.
2. Cross-view transformers fuse the backbone features of all ``N`` views at
   every scale (attention over the view axis at each spatial position).
3. A convolutional decoder predicts per-pixel Gaussian parameters and a refined
   log-depth that is blended with the stereo depth through a learned gate.
4. ``GaussianAdapter`` lifts the per-pixel Gaussians to world space and
   ``aggregate_gaussians`` merges redundant splats with depth-adaptive hashing
   and learned per-splat importance weights.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from stereosplat.models.aggregation import aggregate_gaussians
from stereosplat.models.cross_view_attention import CrossViewTransformer
from stereosplat.models.layers import ResidualBlock
from stereosplat.rendering.gaussians import Gaussians


class StereoSplat(nn.Module):
    """Feed-forward Gaussian splatting from rectified stereo windows.

    Input layout: views are ordered ``[L0, L1, ..., R0, R1, ...]``.

    Input shapes:
        - images: [B, N, 3, H, W] (normalized with the dataset statistics)
        - poses: [B, N, 4, 4] (world-to-camera)
        - intrinsics: [B, N, 3, 3] (pixels)

    Output shapes:
        - depths: [B, N, 1, H, W]
        - raw_gaussians: [B, N, H, W, D] with D = 1 + 3 + 4 + 3 * d_sh
    """

    def __init__(
        self,
        fs_model: nn.Module,
        gaussian_adapter: nn.Module,
        max_depth: float = 80.0,
        merge_distance: float = 0.003,
        appearance_resolution: float = 0.2,
        weight_temperature_init: float = 1.0,
        raw_gaussian_channels: int = 32,
        cross_view_num_heads: int = 4,
        cross_view_num_layers: int = 4,
        cross_view_ffn_ratio: int = 4,
    ) -> None:
        super().__init__()
        self.fs_model = fs_model
        self.gaussian_adapter = gaussian_adapter
        self.max_depth = max_depth
        self.merge_distance = merge_distance
        self.appearance_resolution = appearance_resolution
        self.raw_gaussian_channels = raw_gaussian_channels

        # ---- decoder ----
        self.gaussian_head = nn.Linear(raw_gaussian_channels, 1 + 3 + 4 + 3 * self.d_sh)

        self.image_encoder = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1, bias=False))
        self.depth_encoder_skip = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False)
        )
        self.depth_enc1 = nn.Sequential(
            ResidualBlock(32, 64, stride=1), ResidualBlock(64, 64)
        )
        self.depth_enc2 = nn.Sequential(
            ResidualBlock(64, 64, stride=2), ResidualBlock(64, 64)
        )
        self.depth_enc3 = nn.Sequential(
            ResidualBlock(64, 64, stride=2), ResidualBlock(64, 64)
        )

        self.block_context0 = nn.Sequential(
            ResidualBlock(320, 128), nn.Upsample(scale_factor=2, mode="bilinear")
        )
        self.block_context1 = nn.Sequential(
            ResidualBlock(320, 128), nn.Upsample(scale_factor=2, mode="bilinear")
        )
        self.block_context2 = nn.Sequential(
            ResidualBlock(128, 128), nn.Upsample(scale_factor=2, mode="bilinear")
        )

        self.block_dec0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"), ResidualBlock(96, 64)
        )
        self.block_dec1 = nn.Sequential(ResidualBlock(224, 128), ResidualBlock(128, 64))

        self.block_raw_gaussians = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(128, 64, 3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(64, raw_gaussian_channels, 1),
        )

        self.block_depth_output = nn.Sequential(
            ResidualBlock(128, 32),
            ResidualBlock(32, 32),
            nn.Conv2d(32, 1, 3, padding=1, bias=True),
        )

        # Gated residual so the model starts by preserving the stereo depth.
        # sigmoid(-3) ~= 0.05 -> nearly pure pass-through at init.
        self.depth_residual_gate = nn.Parameter(torch.tensor(-3.0))

        # ---- learned aggregation weights ----
        self.weight_head = nn.Linear(raw_gaussian_channels, 1)
        nn.init.zeros_(self.weight_head.weight)
        nn.init.zeros_(self.weight_head.bias)
        self.log_weight_temperature = nn.Parameter(
            torch.tensor(math.log(weight_temperature_init))
        )
        # Decoder features of the last forward pass, consumed (and cleared) by
        # ``aggregate_gaussians``.
        self._cached_splat_features: torch.Tensor | None = None

        # ---- cross-view feature fusion (all backbone feature scales) ----
        def _cross_view(dim: int) -> CrossViewTransformer:
            return CrossViewTransformer(
                embed_dim=dim,
                num_heads=cross_view_num_heads,
                num_layers=cross_view_num_layers,
                ffn_ratio=cross_view_ffn_ratio,
            )

        self.cross_view_ctx2 = _cross_view(128)
        self.cross_view_ctx1 = _cross_view(128)
        self.cross_view_ctx0 = _cross_view(128)
        self.cross_view_stem = _cross_view(32)

    @property
    def d_sh(self) -> int:
        return (self.gaussian_adapter.sh_degree + 1) ** 2

    def train(self, mode: bool = True) -> StereoSplat:
        """Set train/eval mode; the frozen backbone always stays in eval mode."""
        super().train(mode)
        self.fs_model.eval()
        return self

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Predict per-view depth and per-pixel Gaussian parameters.

        Args:
            data: Dictionary with ``images`` [B, N, 3, H, W], ``poses``
                [B, N, 4, 4] and ``intrinsics`` [B, N, 3, 3].

        Returns:
            Dictionary with ``depths`` [B, N, 1, H, W] and ``raw_gaussians``
            [B, N, H, W, D].
        """
        B = data["images"].shape[0]
        N = data["images"].shape[1]
        num_pairs = N // 2

        left_depths, right_depths, features = self._forward_stereo(
            data["images"], data["poses"], data["intrinsics"], num_pairs
        )
        fs_depth = torch.cat([left_depths, right_depths], dim=1).unsqueeze(2)

        refined_depths, splats = self._forward_decoder(
            data["images"], fs_depth, features, B, N
        )
        return {"depths": refined_depths, "raw_gaussians": splats}

    def _forward_stereo(
        self,
        images: torch.Tensor,
        poses: torch.Tensor,
        intrinsics: torch.Tensor,
        num_pairs: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Stereo depth for all pairs of all batch elements in two backbone calls.

        Returns:
            left_depths:  [B, P, H, W]
            right_depths: [B, P, H, W]
            features:     dict of [B*N, C, h, w] tensors, each batch element's
                          N views contiguous in ``[L..., R...]`` order.
        """
        B = images.shape[0]

        baselines = torch.linalg.norm(
            poses[:, 0, :3, -1] - poses[:, num_pairs, :3, -1], dim=-1
        )  # [B]
        focal_lengths = intrinsics[:, 0, 0, 0]  # [B]
        fb = focal_lengths * baselines  # [B]
        min_disps = fb / self.max_depth  # [B]

        fb_pp = fb[:, None].expand(-1, num_pairs).reshape(-1, 1, 1)  # [B*P, 1, 1]
        md_pp = (
            min_disps[:, None].expand(-1, num_pairs).reshape(-1, 1, 1, 1)
        )  # [B*P, 1, 1, 1]

        left_imgs = images[:, :num_pairs].reshape(-1, *images.shape[2:])  # [B*P, C, H, W]
        right_imgs = images[:, num_pairs:].reshape(-1, *images.shape[2:])
        left_flipped = torch.flip(left_imgs, dims=[3])
        right_flipped = torch.flip(right_imgs, dims=[3])

        with torch.no_grad():
            left_disp, feats_raw = self.fs_model(left_imgs, right_imgs)
            left_disp_c = torch.clamp(left_disp, min=md_pp)
            left_depths = fb_pp / (left_disp_c[:, 0] + 1e-8)  # [B*P, H, W]

            # Right-view disparity: run the horizontally flipped pair.
            right_disp, _ = self.fs_model(
                right_flipped, left_flipped, return_features=False
            )
            right_d = torch.flip(right_disp[:, 0], dims=[2]).clone()
            right_d_c = torch.clamp(
                right_d, min=md_pp[:, 0, 0, 0].unsqueeze(1).unsqueeze(2)
            )
            right_depths = fb_pp / (right_d_c + 1e-8)  # [B*P, H, W]

        H, W = left_depths.shape[-2:]
        left_depths = left_depths.float().view(B, num_pairs, H, W)
        right_depths = right_depths.float().view(B, num_pairs, H, W)

        # Backbone layout [B0L0 B0L1 B1L0 B1L1 | B0R0 B0R1 B1R0 B1R1]
        # -> per-element layout [B0L0 B0L1 B0R0 B0R1 | B1L0 B1L1 B1R0 B1R1]
        features = {}
        for key, feat in feats_raw.items():
            f5 = feat.view(2, B, num_pairs, *feat.shape[1:])
            f5 = f5.permute(1, 0, 2, 3, 4, 5).contiguous()
            features[key] = f5.view(-1, *feat.shape[1:])

        return left_depths, right_depths, features

    def _forward_decoder(
        self,
        images: torch.Tensor,
        fs_depth: torch.Tensor,
        fs_features: dict[str, torch.Tensor],
        B: int,
        N: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cross-view fusion + decoder.

        Args:
            images: [B, N, C, H, W]
            fs_depth: [B, N, 1, H, W]
            fs_features: dict of [B*N, C, h, w] tensors
            B: batch size
            N: views per batch element

        Returns:
            depth: [B, N, 1, H, W]
            raw_gaussian_params: [B, N, H, W, D]
        """
        _, _, C, H, W = images.shape

        ctx2 = self.cross_view_ctx2(fs_features["ctx2"], B, N)
        ctx1 = self.cross_view_ctx1(fs_features["ctx1"], B, N)
        ctx0 = self.cross_view_ctx0(fs_features["ctx0"], B, N)
        stem = self.cross_view_stem(fs_features["stem"], B, N)

        images_flat = images.view(B * N, C, H, W)
        fs_depth_flat = fs_depth.view(B * N, 1, H, W)

        image_features = self.image_encoder(images_flat)
        log_fs_depth = torch.log(fs_depth_flat + 1e-6)
        depth_features = self.depth_encoder_skip(log_fs_depth)

        depth_enc1 = self.depth_enc1(depth_features)
        depth_enc2 = self.depth_enc2(depth_enc1)
        depth_enc3 = self.depth_enc3(depth_enc2)

        x = self.block_context2(ctx2)  # h / 16, w / 16

        x = torch.cat([x, depth_enc3, ctx1], dim=1)
        x = self.block_context1(x)  # h / 8, w / 8

        x = torch.cat([x, depth_enc2, ctx0], dim=1)
        x = self.block_context0(x)  # h / 4, w / 4

        x = torch.cat([x, depth_enc1, stem], dim=1)
        x = self.block_dec1(x)  # h / 2, w / 2

        x = torch.cat([x, depth_features], dim=1)
        x = self.block_dec0(x)

        x = torch.cat([x, image_features], dim=1)

        # Gaussian parameters: [B*N, 32, H, W] -> [B, N, H, W, D]
        splat_features = self.block_raw_gaussians(x)
        self._cached_splat_features = splat_features
        raw_gaussian_params = splat_features.permute(0, 2, 3, 1)  # [B*N, H, W, 32]
        raw_gaussian_params = raw_gaussian_params.reshape(B * N, H * W, -1)
        raw_gaussian_params = self.gaussian_head(raw_gaussian_params)
        raw_gaussian_params = raw_gaussian_params.reshape(B, N, H, W, -1)

        # Depth: gated interpolation between stereo depth and refined depth.
        refined_log_depth = self.block_depth_output(x)
        refined_depth = torch.clamp(
            torch.exp(refined_log_depth) - 1e-3, min=1e-3, max=self.max_depth
        )
        gate = torch.sigmoid(self.depth_residual_gate)
        depth = (1 - gate) * fs_depth_flat + gate * refined_depth
        depth = depth.view(B, N, 1, H, W)

        return depth, raw_gaussian_params

    # ------------------------------------------------------------------
    # Gaussians
    # ------------------------------------------------------------------

    def aggregate_gaussians(self, gaussians: Gaussians) -> Gaussians:
        """Merge the per-pixel Gaussians of the last ``forward`` call.

        Must be called once after each ``forward``: it consumes the decoder
        features cached by that pass to compute per-splat importance weights.
        """
        feat = self._cached_splat_features
        if feat is None:
            raise RuntimeError("aggregate_gaussians() must follow a forward() call.")
        self._cached_splat_features = None

        feat_flat = feat.permute(0, 2, 3, 1).reshape(feat.shape[0], -1, feat.shape[1])
        weight_logits = self.weight_head(feat_flat)  # [B*N, H*W, 1]
        B = gaussians.means.shape[0]
        weights = weight_logits.reshape(B, -1, 1)
        temperature = self.log_weight_temperature.exp()

        return aggregate_gaussians(
            gaussians,
            weights,
            temperature,
            self.merge_distance,
            self.appearance_resolution,
        )

    def forward_gaussians(self, data: dict[str, torch.Tensor]) -> Gaussians:
        """End-to-end: predict -> lift to world space -> merge."""
        outputs = self.forward(data)
        gaussians = self.gaussian_adapter(
            extrinsics=data["poses"],
            intrinsics=data["intrinsics"],
            depths=outputs["depths"],
            raw_gaussians=outputs["raw_gaussians"],
        )
        return self.aggregate_gaussians(gaussians)
