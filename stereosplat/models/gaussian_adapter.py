# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified by Meta Platforms, Inc. and affiliates for StereoSplat.
# Derived from Depth Anything 3 (https://github.com/ByteDance-Seed/Depth-Anything-3).

from __future__ import annotations

from typing import Optional

import torch
from einops import einsum, rearrange, repeat
from torch import nn
from stereosplat.rendering.gaussians import Gaussians
from stereosplat.utils.geometry import (
    affine_inverse,
    cam_quat_xyzw_to_world_quat_wxyz,
    sample_image_grid,
)
from stereosplat.utils.sh import rotate_sh


class GaussianAdapter(nn.Module):
    sh_mask: torch.Tensor

    def __init__(
        self,
        sh_degree: int = 0,
        pred_offset_xy: bool = True,
        gaussian_scale_min: float = 1e-5,
        gaussian_scale_max: float = 30.0,
    ) -> None:
        super().__init__()
        self.sh_degree = sh_degree
        self.pred_offset_xy = pred_offset_xy
        self.gaussian_scale_min = gaussian_scale_min
        self.gaussian_scale_max = gaussian_scale_max

        # Create a mask for the spherical harmonics coefficients. This ensures that at
        # initialization, the coefficients are biased towards having a large DC
        # component and small view-dependent components.
        self.register_buffer(
            "sh_mask",
            torch.ones((self.d_sh,), dtype=torch.float32),
            persistent=False,
        )
        for degree in range(1, sh_degree + 1):
            self.sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree

    def forward(
        self,
        extrinsics: torch.Tensor,  # "*#batch 4 4"
        intrinsics: torch.Tensor,  # "*#batch 3 3"
        depths: torch.Tensor,  # "*#batch"
        raw_gaussians: torch.Tensor,  # "*#batch _"
        scores: Optional[torch.Tensor] = None,
        top_k: int = 2048,
    ) -> Gaussians:
        device = extrinsics.device
        dtype = raw_gaussians.dtype
        _, _, _, H, W = depths.shape
        b, v = raw_gaussians.shape[:2]

        # get cam2worlds and intr_normed to adapt to 3DGS codebase
        cam2worlds = affine_inverse(extrinsics)
        intr_normed = intrinsics.clone().detach()
        intr_normed[..., 0, :] /= W
        intr_normed[..., 1, :] /= H

        if scores is not None:
            # Flatten inputs
            flat_mask = scores.view(b * v, -1)
            flat_depths = depths.view(b * v, -1)
            flat_gaussians = raw_gaussians.view(b * v, H * W, -1)

            # Select Top-K
            top_scores, indices = torch.topk(flat_mask, k=top_k, dim=1)

            # Gather Data
            gs_depths = torch.gather(flat_depths, 1, indices).view(b, v, 1, top_k)

            # Expand indices for gaussians
            idx_gauss = indices.unsqueeze(-1).expand(-1, -1, flat_gaussians.shape[-1])
            selected_raw = torch.gather(flat_gaussians, 1, idx_gauss)

            # --- FIX: Avoid In-Place Modification ---
            # 1. Split opacity from the rest
            opacity = selected_raw[..., 0:1]  # Keep dim
            other_attrs = selected_raw[..., 1:]

            # 2. Modulate Opacity (Creates a NEW tensor, safe for autograd)
            # top_scores is (BV, K), opacity is (BV, K, 1)
            modulated_opacity = opacity * top_scores.unsqueeze(-1)

            # 3. Concatenate back together
            raw_gaussians = torch.cat([modulated_opacity, other_attrs], dim=-1)
            # ----------------------------------------

            # Reshape
            raw_gaussians = raw_gaussians.view(b, v, 1, top_k, -1)

            # Compute Coords
            xy_ray = torch.stack(
                [((indices % W).float() + 0.5) / W, ((indices // W).float() + 0.5) / H],
                dim=-1,
            ).view(b, v, 1, top_k, 2)

            H_grid, W_grid = 1, top_k
        else:
            gs_depths = depths[:, :, 0]
            xy_ray, _ = sample_image_grid((H, W), device)
            xy_ray = xy_ray[None, None, ...].expand(b, v, -1, -1, -1)
            H_grid, W_grid = H, W

        # 1.4) unproject depth + xy to world ray
        c2w_broadcast = cam2worlds.unsqueeze(-3).unsqueeze(-3)
        intr_inv_broadcast = intr_normed.float().inverse().unsqueeze(-3).unsqueeze(-3)
        xy_hom = torch.cat([xy_ray, torch.ones_like(xy_ray[..., :1])], dim=-1)
        cam_rays = torch.matmul(intr_inv_broadcast, xy_hom.unsqueeze(-1)).squeeze(-1)
        cam_points = cam_rays * gs_depths.unsqueeze(-1)
        R = c2w_broadcast[..., :3, :3]  # (b, v, 1, 1, 3, 3)
        T = c2w_broadcast[..., :3, 3]  # (b, v, 1, 1, 3)
        gs_means_world = torch.matmul(R, cam_points.unsqueeze(-1)).squeeze(-1) + T
        gs_means_world = rearrange(gs_means_world, "b v h w d -> b (v h w) d")

        # 2. compute other GS attributes
        opacities, scales, rotations, sh = raw_gaussians.split(
            (1, 3, 4, 3 * self.d_sh), dim=-1
        )

        # 2.1) 3DGS scales
        # make the scale invarient to resolution
        scale_min = self.gaussian_scale_min
        scale_max = self.gaussian_scale_max
        scales = scale_min + (scale_max - scale_min) * scales.sigmoid()
        pixel_size = torch.tensor((W, H), dtype=dtype, device=device).reciprocal()
        multiplier = self.get_scale_multiplier(intr_normed, pixel_size)
        gs_scales = scales * gs_depths[..., None] * multiplier[..., None, None, None]
        gs_scales = rearrange(gs_scales, "b v h w d -> b (v h w) d")

        # 2.2) 3DGS quaternion (world space)
        # due to historical issue, assume quaternion in order xyzw, not wxyz
        # Normalize the quaternion features to yield a valid quaternion.
        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + 1e-8)
        # rotate them to world space
        cam_quat_xyzw = rearrange(rotations, "b v h w c -> b (v h w) c")
        c2w_mat = repeat(
            cam2worlds,
            "b v i j -> b (v h w) i j",
            h=H_grid,
            w=W_grid,
        )
        world_quat_wxyz = cam_quat_xyzw_to_world_quat_wxyz(cam_quat_xyzw, c2w_mat)
        gs_rotations_world = world_quat_wxyz  # b (v h w) c

        # 2.3) 3DGS color / SH coefficient (world space)
        sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
        sh = sh * self.sh_mask

        gs_sh_world = rotate_sh(sh, cam2worlds[:, :, None, None, None, :3, :3])
        gs_sh_world = rearrange(gs_sh_world, "b v h w xyz d_sh -> b (v h w) xyz d_sh")

        # 2.4) 3DGS opacity
        opacities = opacities.sigmoid()
        gs_opacities = rearrange(opacities, "b v h w ... -> b (v h w) ...")

        return Gaussians(
            means=gs_means_world,
            harmonics=gs_sh_world,
            opacities=gs_opacities[..., 0],
            scales=gs_scales,
            rotations=gs_rotations_world,
        )

    def get_scale_multiplier(
        self,
        intrinsics: torch.Tensor,  # "*#batch 3 3"
        pixel_size: torch.Tensor,  # "*#batch 2"
        multiplier: float = 0.1,
    ) -> torch.Tensor:  # " *batch"
        xy_multipliers = multiplier * einsum(
            intrinsics[..., :2, :2].float().inverse().to(intrinsics),
            pixel_size,
            "... i j, j -> ... i",
        )
        return xy_multipliers.sum(dim=-1)

    @property
    def d_sh(self) -> int:
        return (self.sh_degree + 1) ** 2
