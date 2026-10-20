# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Rendering visualization utilities for TensorBoard image logging.

These functions create comparison grids of rendered vs ground-truth images
and depths, suitable for logging to TensorBoard.

NOTE: This module intentionally avoids importing open3d to prevent
duplicate-entry build conflicts. Point cloud visualization functions
remain in vis_utils.py.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.cm as cm
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def to_numpy_img(tensor: torch.Tensor | np.ndarray) -> np.ndarray:
    """
    Helper to convert (C, H, W) or (H, W, C) tensor to (H, W, 3) uint8 numpy array.
    Assumes input is in [0, 1].
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu()
    else:
        tensor = torch.tensor(tensor)

    if tensor.ndim == 3 and tensor.shape[0] in [1, 3]:
        tensor = tensor.permute(1, 2, 0)

    if tensor.shape[-1] == 1:
        tensor = tensor.repeat(1, 1, 3)

    img = tensor.numpy()
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    return img


def apply_colormap(tensor, cmap_name="turbo"):
    """
    Applies a matplotlib colormap to a (1, H, W) or (H, W) tensor.
    Returns a (H, W, 3) uint8 numpy array.
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()

    # Remove channel dim if present (1, H, W) -> (H, W)
    if tensor.ndim == 3:
        tensor = tensor.squeeze(0)

    cmap = cm.get_cmap(cmap_name)

    # Map [0, 1] to RGBA, drop Alpha, convert to uint8
    colored = cmap(tensor)[:, :, :3]  # (H, W, 3) float
    colored = (colored * 255).astype(np.uint8)
    return colored


def add_header(
    grid_img: Image.Image, col_names: Sequence[str], col_width: int
) -> Image.Image:
    """Adds a text header row above the image grid."""
    header_height = 30
    header_img = Image.new(
        "RGB", (grid_img.width, header_height), color=(255, 255, 255)
    )
    draw = ImageDraw.Draw(header_img)

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    for i, name in enumerate(col_names):
        x_pos = i * col_width + 10  # small padding
        draw.text((x_pos, 5), name, fill=(0, 0, 0), font=font)

    final_h = grid_img.height + header_height
    final_img = Image.new("RGB", (grid_img.width, final_h))
    final_img.paste(header_img, (0, 0))
    final_img.paste(grid_img, (0, header_height))
    return final_img


def visualize_sample(
    renderer_output: dict,
    target_output: dict,
    output_path: str | None = None,
    vis_size: tuple[int, int] | None = (320, 640),
) -> Image.Image:
    """Create a comparison grid of rendered vs ground-truth images and depths.

    Args:
        renderer_output: Dict with "colors" [B,V,C,H,W] and "depths" [B,V,H,W].
        target_output: Dict with "original_images" [B,V,C,H,W] and "depths" [B,V,1,H,W].
        output_path: Optional path to save the image.

    Returns:
        PIL Image with columns: Render Color, GT Color, Color Diff (x10),
        Render Depth, GT Depth, Depth Diff.
    """

    def flatten_bn(x: torch.Tensor) -> torch.Tensor:
        if x.dim() >= 4:
            return x.reshape(-1, *x.shape[2:])
        return x

    rend_color = renderer_output["colors"]
    tgt_color = target_output["original_images"]
    rend_depth = renderer_output["depths"]
    tgt_depth = target_output["depths"]

    rend_color = flatten_bn(rend_color).detach().cpu()
    tgt_color = flatten_bn(tgt_color).detach().cpu()
    rend_depth = flatten_bn(rend_depth).detach().cpu()
    tgt_depth = flatten_bn(tgt_depth).detach().cpu()

    if vis_size is not None:
        import torch.nn.functional as F

        h, w = vis_size
        rend_color = F.interpolate(
            rend_color, size=(h, w), mode="bilinear", align_corners=False
        )
        tgt_color = F.interpolate(
            tgt_color, size=(h, w), mode="bilinear", align_corners=False
        )
        if rend_depth.ndim == 3:
            rend_depth = rend_depth.unsqueeze(1)
        if tgt_depth.ndim == 3:
            tgt_depth = tgt_depth.unsqueeze(1)
        rend_depth = F.interpolate(rend_depth, size=(h, w), mode="nearest").squeeze(1)
        tgt_depth = F.interpolate(tgt_depth, size=(h, w), mode="nearest").squeeze(1)

    rows = []
    num_samples = rend_color.shape[0]

    for i in range(num_samples):
        rc = rend_color[i]
        tc = tgt_color[i]

        c_diff = torch.abs(rc - tc).mean(dim=0, keepdim=True) * 10

        rd = rend_depth[i]
        td = tgt_depth[i]

        if rd.ndim == 2:
            rd = rd.unsqueeze(0)
        if td.ndim == 2:
            td = td.unsqueeze(0)
        if rd.shape[0] > 1:
            rd = rd[0:1]
        if td.shape[0] > 1:
            td = td[0:1]

        d_diff = torch.abs(rd - td)
        d_diff_vis = d_diff / (d_diff.max() + 1e-8)

        img_rc = to_numpy_img(rc)
        img_tc = to_numpy_img(tc)
        img_cdiff = apply_colormap(c_diff, cmap_name="magma")

        rd_vis = rd / (rd.max() + 1e-8)
        td_vis = td / (td.max() + 1e-8)
        img_rd = apply_colormap(rd_vis, cmap_name="magma")
        img_td = apply_colormap(td_vis, cmap_name="magma")

        img_ddiff = apply_colormap(d_diff_vis, cmap_name="magma")

        img_list = [img_rc, img_tc, img_cdiff, img_rd, img_td, img_ddiff]
        row_img = np.concatenate(img_list, axis=1)
        rows.append(row_img)

    grid_array = np.concatenate(rows, axis=0)
    grid_pil = Image.fromarray(grid_array)

    col_width = rend_color.shape[-1]
    column_names = [
        "Render Color",
        "GT Color",
        "Color Diff (x10)",
        "Render Depth",
        "GT Depth",
        "Depth Diff",
    ]

    final_pil = add_header(grid_pil, column_names, col_width)

    if output_path is not None:
        final_pil.save(output_path)

    return final_pil
