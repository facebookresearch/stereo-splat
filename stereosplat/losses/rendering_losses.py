# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import itertools
from math import exp

import torch
import torch.nn as nn
import torch.nn.functional as F
from lpips import LPIPS
from torch.autograd import Variable

# The SSIM helpers below (gaussian, create_window, ssim, _ssim) are adapted from
# pytorch-ssim (https://github.com/Po-Hsun-Su/pytorch-ssim), MIT License.
# See third_party/README.md.


def gaussian(window_size: int, sigma: float) -> torch.Tensor:
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size: int, channel: int) -> torch.Tensor:
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    size_average: bool = True,
) -> torch.Tensor:
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window: torch.Tensor,
    window_size: int,
    channel: int,
    size_average: bool = True,
) -> torch.Tensor:
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


# ------------------------------------------------------------------
# Helpers for LPIPS
# ------------------------------------------------------------------


def _convert_to_buffer(module: nn.Module, persistent: bool = True) -> None:
    """Convert all parameters in *module* to buffers (non-trainable)."""
    for _, child in list(module.named_children()):
        _convert_to_buffer(child, persistent)

    for name, parameter_or_buffer in list(
        itertools.chain(
            module.named_parameters(recurse=False),
            module.named_buffers(recurse=False),
        )
    ):
        value = parameter_or_buffer.detach().clone()
        delattr(module, name)
        module.register_buffer(name, value, persistent=persistent)


# ------------------------------------------------------------------
# Loss
# ------------------------------------------------------------------


class RenderingLoss(nn.Module):
    """Combined rendering loss with independently weighted L1, SSIM, LPIPS, and depth.

    Config keys (all optional, shown with defaults):
        color_l1_weight:    0.8   - weight for pixel-wise L1
        color_ssim_weight:  0.2   - weight for DSSIM (1 - SSIM)
        color_lpips_weight: 0.0   - weight for perceptual LPIPS loss
        depth_weight:       1.0   - weight for masked depth L1

    Backward-compatible: old configs using ``color_weight`` + ``lambda_dssim``
    are automatically converted to the new scheme.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        self.depth_weight = config.get("depth_weight", 1.0)

        # --- Backward compat: old color_weight + lambda_dssim scheme ---
        if "color_weight" in config and "color_l1_weight" not in config:
            cw = config["color_weight"]
            ld = config.get("lambda_dssim", 0.2)
            self.color_l1_weight = cw * (1.0 - ld)
            self.color_ssim_weight = cw * ld
        else:
            self.color_l1_weight = config.get("color_l1_weight", 0.8)
            self.color_ssim_weight = config.get("color_ssim_weight", 0.2)

        self.color_lpips_weight = config.get("color_lpips_weight", 0.0)

        # --- LPIPS network (only instantiate when needed) ---
        self.lpips_net: LPIPS | None = None
        if self.color_lpips_weight > 0:
            lpips_net = LPIPS(net="vgg")
            _convert_to_buffer(lpips_net, persistent=False)
            self.lpips_net = lpips_net

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
