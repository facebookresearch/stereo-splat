# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
import torch.nn as nn
from pytorch_msssim import ms_ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


_DEFAULT_DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


class RenderingEvaluator(nn.Module):
    def __init__(self, device: str | torch.device = _DEFAULT_DEVICE) -> None:
        super().__init__()
        self.device = device
        self.psnr_data_range = 1.0
        self.msssim_win_size = 11
        self.msssim_data_range = 1.0
        self.lpips = LearnedPerceptualImagePatchSimilarity(
            net_type="alex", normalize=True
        ).to(self.device)

    def __call__(self, pred: dict, target: dict) -> dict:  # pyre-ignore[29]
        return self.forward(pred, target)

    def compute_psnr(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        diff = (img1 - img2) ** 2
        if mask is not None:
            if mask.shape[1] != img1.shape[1]:
                mask = mask.expand_as(img1)
            # Per-image masked MSE -> per-image PSNR -> average
            B = img1.shape[0]
            psnr_values = []
            for i in range(B):
                valid = diff[i][mask[i] > 0]
                if valid.numel() == 0:
                    psnr_values.append(torch.tensor(0.0, device=self.device))
                    continue
                mse_i = torch.clamp(valid.mean(), min=1e-8)
                psnr_values.append(
                    20 * torch.log10(self.psnr_data_range / torch.sqrt(mse_i))
                )
            return torch.stack(psnr_values).mean()
        else:
            # Per-image MSE -> per-image PSNR -> average
            mse_per_image = diff.view(img1.shape[0], -1).mean(1)
            mse_per_image = torch.clamp(mse_per_image, min=1e-8)
            psnr_per_image = 20 * torch.log10(
                self.psnr_data_range / torch.sqrt(mse_per_image)
            )
            return psnr_per_image.mean()

    def compute_msssim(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # NOTE: We deliberately do NOT mask the images for MS-SSIM.
        #
        # Previously this method multiplied both `img1` and `img2` by `mask`
        # before computing MS-SSIM, with the intent of "ignoring invalid
        # regions". For sparse masks (e.g. KITTI's sparse-LiDAR depth-validity
        # mask covers only ~5-10 % of pixels), this zeros out ~95 % of *both*
        # images identically, and MS-SSIM's local 11x11 windows then evaluate
        # to 1.0 on the constant-zero regions, dragging the global score to
        # ~0.99 regardless of how wrong the actual rendered region is.
        #
        # Standard NVS evaluation (e.g. AnySplat / pixelSplat / MVSplat /
        # DepthSplat papers, and skimage `structural_similarity`) computes
        # SSIM on full unmasked images. We do the same. PSNR, where masking
        # is mathematically meaningful (per-pixel MSE averaged over valid
        # pixels), still honours the mask in `compute_psnr`.
        del mask  # intentionally unused; kept in signature for API stability
        return ms_ssim(
            img1,
            img2,
            data_range=self.msssim_data_range,
            size_average=True,
            win_size=self.msssim_win_size,
        )

    def forward(self, pred: dict, target: dict) -> dict:
        """
        Expects keys 'masks' in inputs if masking is desired.
        masks should be (B, 1, H, W) or (B, V, 1, H, W)
        """
        # Extract masks if available, otherwise None
        mask = target.get("masks", None)

        # If mask exists but is all-zero (e.g. dataset has no depth),
        # skip masking for color metrics and skip depth metrics entirely.
        # When mask is None (not provided), always compute depth metrics.
        has_valid_depth = mask is None or mask.sum() > 0

        color_metrics = self.compute_color_metrics(
            pred_color=torch.clamp(pred["colors"], min=0.0, max=1.0),
            target_color=target["original_images"],
            mask=mask if has_valid_depth else None,
        )
        if has_valid_depth:
            depth_metrics = self.compute_depth_metrics(
                pred_depth=pred["depths"],
                target_depth=target["depths"][:, :, 0],
                mask=mask,
                intrinsics=target.get("intrinsics", None),
                poses=target.get("poses", None),
            )
        else:
            depth_metrics = {}
        return {**color_metrics, **depth_metrics}

    def compute_color_metrics(
        self,
        pred_color: torch.Tensor,
        target_color: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict[str, float]:
        B, V, C, H, W = pred_color.shape
        pred_color = pred_color.view(B * V, C, H, W).to(self.device)
        target_color = target_color.view(B * V, C, H, W).to(self.device)

        if mask is not None:
            mask = mask.view(B * V, 1, H, W).to(self.device)

        # PSNR is averaged only over masked-valid pixels (mathematically
        # meaningful: per-pixel MSE over the valid region). MS-SSIM and
        # LPIPS, in contrast, are *spatial* perceptual metrics that compute
        # local statistics over a window: zeroing out invalid pixels in both
        # images creates artificial constant-zero regions that bias the
        # metric (toward 1.0 for SSIM, toward an arbitrary near-zero
        # constant for LPIPS), especially for sparse masks like KITTI's
        # LiDAR-derived depth-validity mask. We therefore compute SSIM and
        # LPIPS on the full unmasked images, matching AnySplat / pixelSplat
        # / MVSplat / DepthSplat conventions.
        psnr_val = self.compute_psnr(pred_color, target_color, mask).item()
        msssim_val = self.compute_msssim(pred_color, target_color).item()
        lpips_val = self.lpips(pred_color, target_color).item()

        return {"psnr": psnr_val, "sssim": msssim_val, "lpips": lpips_val}

    def compute_depth_metrics(
        self,
        pred_depth: torch.Tensor,
        target_depth: torch.Tensor,
        mask: torch.Tensor | None = None,
        **kwargs: torch.Tensor | None,
    ) -> dict[str, float]:
        pred_depth = pred_depth.float().to(self.device)
        target_depth = target_depth.float().to(self.device)

        diff = torch.abs(pred_depth - target_depth)

        if mask is not None:
            mask = mask.view_as(pred_depth).to(self.device)
            valid = (mask > 0) & (target_depth > 1e-8)
        else:
            valid = target_depth > 1e-8

        if valid.sum() == 0:
            return {"l1": 0.0, "absrel": 0.0, "delta1": 0.0}

        l1_score = torch.mean(diff[valid]).item()

        # AbsRel: mean(|pred - gt| / gt)
        absrel = torch.mean(diff[valid] / target_depth[valid]).item()

        # delta1: % pixels where max(pred/gt, gt/pred) < 1.25
        pred_valid = pred_depth[valid].clamp(min=1e-8)
        gt_valid = target_depth[valid]
        ratio = torch.max(pred_valid / gt_valid, gt_valid / pred_valid)
        delta1 = (ratio < 1.25).float().mean().item() * 100.0

        return {"l1": l1_score, "absrel": absrel, "delta1": delta1}
