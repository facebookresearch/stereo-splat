# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from lpips import LPIPS

# SSIM (Wang et al., "Image quality assessment: from error visibility to
# structural similarity", IEEE TIP 2004) with an isotropic Gaussian weighting
# window, zero padding (output has the input size) and constants
# C1 = (0.01 L)^2, C2 = (0.03 L)^2 for a dynamic range L = 1.
#
# This is our own implementation of the formula above, written for the training
# loss (it reproduces the loss values the paper model was trained with). It is
# not interchangeable with other SSIM implementations:
# - pytorch_msssim.ssim (the dependency used for evaluation) convolves without
#   padding, so it averages only over windows that lie fully inside the image;
#   near the borders the two give different values, and so do the losses and
#   gradients. Use it only if bitwise loss parity with the paper runs does not
#   matter.
# - The SSIM metric reported in the paper is the multi-scale MS-SSIM of
#   pytorch_msssim (stereosplat.evaluation), not this single-scale SSIM.
SSIM_WINDOW_SIZE: int = 11
SSIM_SIGMA: float = 1.5
SSIM_C1: float = 0.01**2
SSIM_C2: float = 0.03**2


def gaussian_window(
    size: int,
    sigma: float,
    channels: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Normalized 2-D Gaussian kernel as a depthwise conv weight [C, 1, k, k]."""
    offsets = torch.arange(size, dtype=torch.float64) - (size // 2)
    profile = torch.exp(-(offsets**2) / (2.0 * sigma**2)).to(torch.float32)
    profile = profile / profile.sum()
    kernel = torch.outer(profile, profile)
    return kernel.expand(channels, 1, size, size).to(device=device, dtype=dtype)


def ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = SSIM_WINDOW_SIZE,
    size_average: bool = True,
) -> torch.Tensor:
    """Structural similarity of two image batches ``[N, C, H, W]``.

    Local statistics are Gaussian-weighted (sigma 1.5) per channel. Returns the
    mean SSIM over all pixels, channels and images (``size_average=True``) or
    one value per image (``size_average=False``).
    """
    channels = img1.shape[-3]
    weight = gaussian_window(
        window_size, SSIM_SIGMA, channels, dtype=img1.dtype, device=img1.device
    )

    def local_mean(x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, weight, padding=window_size // 2, groups=channels)

    mean_x = local_mean(img1)
    mean_y = local_mean(img2)
    mean_xx = mean_x**2
    mean_yy = mean_y**2
    mean_xy = mean_x * mean_y
    var_x = local_mean(img1 * img1) - mean_xx
    var_y = local_mean(img2 * img2) - mean_yy
    cov_xy = local_mean(img1 * img2) - mean_xy

    luminance = 2.0 * mean_xy + SSIM_C1
    structure = 2.0 * cov_xy + SSIM_C2
    norm = (mean_xx + mean_yy + SSIM_C1) * (var_x + var_y + SSIM_C2)
    ssim_map = luminance * structure / norm

    if size_average:
        return ssim_map.mean()
    per_pixel = ssim_map.mean(dim=1)  # average over channels first
    return per_pixel.mean(dim=1).mean(dim=1)


# ------------------------------------------------------------------
# Loss
# ------------------------------------------------------------------


class RenderingLoss(nn.Module):
    """Combined rendering loss with independently weighted L1, SSIM, LPIPS, and depth.

    Config keys (all optional, shown with defaults):
        color_l1_weight:    0.8   - weight for pixel-wise L1
        color_ssim_weight:  0.2   - weight for DSSIM (1 - SSIM)
        color_lpips_weight: 0.0   - weight for perceptual LPIPS loss (VGG)
        depth_weight:       1.0   - weight for masked depth L1
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        self.color_l1_weight = config.get("color_l1_weight", 0.8)
        self.color_ssim_weight = config.get("color_ssim_weight", 0.2)
        self.color_lpips_weight = config.get("color_lpips_weight", 0.0)
        self.depth_weight = config.get("depth_weight", 1.0)

        # Frozen LPIPS network, only built when the term is used. It is a
        # submodule of the loss (moved by .to(device)), not of the model, so it
        # is neither optimized nor stored in model checkpoints.
        self.lpips_net: LPIPS | None = None
        if self.color_lpips_weight > 0:
            self.lpips_net = LPIPS(net="vgg").requires_grad_(False).eval()

    def train(self, mode: bool = True) -> RenderingLoss:
        """Set train/eval mode; the LPIPS network always stays in eval mode."""
        super().train(mode)
        if self.lpips_net is not None:
            self.lpips_net.eval()
        return self

    def forward(self, renderer_output: dict, gt_data: dict) -> dict[str, torch.Tensor]:
        rendered_color = renderer_output["colors"]
        gt_color = gt_data["original_images"]

        # --- Individual color losses ---
        l1_loss = F.l1_loss(rendered_color, gt_color)

        _, _, channels, height, width = rendered_color.shape
        flat_pred = rendered_color.view(-1, channels, height, width)
        flat_gt = gt_color.view(-1, channels, height, width)
        ssim_val = ssim(flat_pred, flat_gt)
        ssim_loss = 1.0 - ssim_val

        if self.lpips_net is not None:
            lpips_loss = self.lpips_net(
                torch.clamp(flat_pred, 0.0, 1.0),
                torch.clamp(flat_gt, 0.0, 1.0),
                normalize=True,
            ).mean()
        else:
            lpips_loss = torch.zeros(1, device=rendered_color.device)[0]

        color_loss = (
            self.color_l1_weight * l1_loss
            + self.color_ssim_weight * ssim_loss
            + self.color_lpips_weight * lpips_loss
        )

        # --- Depth loss ---
        depth_loss = self.depth_loss(
            renderer_output["depths"],
            gt_data["depths"][:, :, 0],
            gt_data["masks"][:, :, 0],
        )

        total_loss = color_loss + self.depth_weight * depth_loss

        return {
            "total_loss": total_loss,
            "color_loss": color_loss,
            "depth_loss": depth_loss,
            "color_l1_loss": l1_loss,
            "color_ssim_loss": ssim_loss,
            "color_lpips_loss": lpips_loss,
        }

    def depth_loss(
        self, rendered_depth: torch.Tensor, gt_depth: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        valid = mask.bool()
        if not valid.any():
            return torch.zeros(1, device=rendered_depth.device, requires_grad=True)[0]
        return F.l1_loss(rendered_depth[valid], gt_depth[valid])
