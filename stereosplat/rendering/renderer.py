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

import math
from math import isqrt
from typing import Literal

import torch
from einops import rearrange, repeat
from gsplat import rasterization
from tqdm import tqdm

from stereosplat.rendering.gaussians import Gaussians
from stereosplat.utils.geometry import (
    affine_inverse,
    as_homogeneous,
    get_fov,
)


def render_3dgs(
    extrinsics: torch.Tensor,  # "batch_views 4 4", w2c
    intrinsics: torch.Tensor,  # "batch_views 3 3", normalized
    image_shape: tuple[int, int],
    gaussian: Gaussians,
    background_color: torch.Tensor | None = None,  # "batch_views 3"
    use_sh: bool = True,
    num_view: int = 1,
    color_mode: Literal["RGB+D", "RGB+ED"] = "RGB+D",
) -> tuple[
    torch.Tensor,  # color, "batch_views 3 height width"
    torch.Tensor,  # depth, "batch_views height width"
    torch.Tensor,  # alpha, "batch_views height width"
]:
    # extract gaussian params
    gaussian_means = gaussian.means
    gaussian_scales = gaussian.scales
    gaussian_quats = gaussian.rotations
    gaussian_opacities = gaussian.opacities
    gaussian_sh_coefficients = gaussian.harmonics
    b, _, _ = extrinsics.shape

    if background_color is None:
        background_color = repeat(torch.tensor([0.0, 0.0, 0.0]), "c -> b c", b=b).to(
            gaussian_sh_coefficients
        )

    degree: int | None = None
    if use_sh:
        _, _, _, n = gaussian_sh_coefficients.shape
        degree = isqrt(n) - 1
        shs = rearrange(gaussian_sh_coefficients, "b g xyz n -> b g n xyz").contiguous()
    else:  # use color
        shs = (
            gaussian_sh_coefficients.squeeze(-1).sigmoid().contiguous()
        )  # (b, g, c), normed to (0, 1)

    h, w = image_shape

    fov_x, fov_y = get_fov(intrinsics).unbind(dim=-1)
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()
    focal_length_x = w / (2 * tan_fov_x)
    focal_length_y = h / (2 * tan_fov_y)

    # Ensure view matrix is float32
    view_matrix = extrinsics.float()

    all_images: list[torch.Tensor] = []
    all_radii: list[torch.Tensor] = []
    all_depths: list[torch.Tensor] = []
    all_alphas: list[torch.Tensor] = []
    # render view in a batch based, each batch contains one scene
    # assume the Gaussian parameters are originally repeated along the view dim
    batch_scene = b // num_view

    def index_i_gs_attr(full_attr: torch.Tensor, idx: int) -> torch.Tensor:
        # return rearrange(full_attr, "(b v) ... -> b v ...", v=num_view)[idx, 0]
        return full_attr[idx]

    for i in range(batch_scene):
        K = torch.zeros(
            (num_view, 3, 3), dtype=torch.float32, device=gaussian_means.device
        )
        K[:, 2, 2] = 1.0

        i_intrinsics = rearrange(intrinsics, "(b v) ... -> b v ...", v=num_view)[i]
        K[:, 0, 0] = focal_length_x.reshape(batch_scene, num_view)[i].float()
        K[:, 1, 1] = focal_length_y.reshape(batch_scene, num_view)[i].float()
        K[:, 0, 2] = (i_intrinsics[:, 0, 2] * w).float()
        K[:, 1, 2] = (i_intrinsics[:, 1, 2] * h).float()

        # --- FIX: Explicitly cast geometric params to .float() ---
        i_means = index_i_gs_attr(gaussian_means, i).float()
        i_scales = index_i_gs_attr(gaussian_scales, i).float()
        i_quats = index_i_gs_attr(gaussian_quats, i).float()
        i_opacities = index_i_gs_attr(gaussian_opacities, i).float()

        # Colors can usually stay as is, or cast to float if SH requires it
        i_colors = index_i_gs_attr(shs, i)

        i_viewmats = rearrange(view_matrix, "(b v) ... -> b v ...", v=num_view)[
            i
        ]  # Already float
        i_backgrounds = rearrange(background_color, "(b v) ... -> b v ...", v=num_view)[
            i
        ]

        render_colors, render_alphas, info = rasterization(
            means=i_means,
            quats=i_quats,
            scales=i_scales,
            opacities=i_opacities,
            colors=i_colors,
            viewmats=i_viewmats,
            Ks=K,
            backgrounds=i_backgrounds,
            render_mode=color_mode,
            width=w,
            height=h,
            packed=False,
            sh_degree=degree if use_sh else None,
        )
        depth = render_colors[..., -1].unbind(dim=0)

        image = rearrange(render_colors[..., :3], "v h w c -> v c h w").unbind(dim=0)
        radii = info["radii"].unbind(dim=0)
        alphas = render_alphas[..., 0].unbind(dim=0)
        try:
            info["means2d"].retain_grad()  # [1, N, 2]
        except Exception:
            pass
        all_images.extend(image)
        all_depths.extend(depth)
        all_radii.extend(radii)
        all_alphas.extend(alphas)

    return torch.stack(all_images), torch.stack(all_depths), torch.stack(all_alphas)


def run_renderer_in_chunk_w_trj_mode(
    gaussians: Gaussians,
    extrinsics: torch.Tensor,  # world2cam, "batch view 4 4" | "batch view 3 4"
    intrinsics: torch.Tensor,  # unnormed intrinsics, "batch view 3 3"
    image_shape: tuple[int, int],
    chunk_size: int | None = 8,
    input_shape: tuple[int, int] | None = None,
    enable_tqdm: bool | None = False,
) -> dict[str, torch.Tensor]:
    cam2world = affine_inverse(as_homogeneous(extrinsics))
    if input_shape is not None:
        in_h, in_w = input_shape
    else:
        in_h, in_w = image_shape
    intr_normed = intrinsics.clone().detach()
    intr_normed[..., 0, :] /= in_w
    intr_normed[..., 1, :] /= in_h
    tgt_c2w = cam2world
    tgt_intr = intr_normed

    _, v = tgt_c2w.shape[:2]
    tgt_extr = affine_inverse(tgt_c2w)
    if chunk_size is None:
        chunk_size = v
    chunk_size = min(v, chunk_size)
    all_colors = []
    all_depths = []
    all_alphas = []
    for chunk_idx in tqdm(
        range(math.ceil(v / chunk_size)),
        desc="Rendering novel views",
        disable=(not enable_tqdm),
        leave=False,
    ):
        s = int(chunk_idx * chunk_size)
        e = int((chunk_idx + 1) * chunk_size)
        cur_n_view = tgt_extr[:, s:e].shape[1]
        color, depth, alpha = render_3dgs(
            extrinsics=rearrange(tgt_extr[:, s:e], "b v ... -> (b v) ..."),  # w2c
            intrinsics=rearrange(tgt_intr[:, s:e], "b v ... -> (b v) ..."),  # normed
            image_shape=image_shape,
            gaussian=gaussians,
            num_view=cur_n_view,
        )
        all_colors.append(rearrange(color, "(b v) ... -> b v ...", v=cur_n_view))
        all_depths.append(rearrange(depth, "(b v) ... -> b v ...", v=cur_n_view))
        all_alphas.append(rearrange(alpha, "(b v) ... -> b v ...", v=cur_n_view))

    all_colors = torch.cat(all_colors, dim=1)
    all_depths = torch.cat(all_depths, dim=1)
    all_alphas = torch.cat(all_alphas, dim=1)

    return {
        "colors": all_colors,
        "depths": all_depths,
        "alphas": all_alphas,
    }
