# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Depth-adaptive learnable aggregation of per-pixel Gaussians.

Per-pixel Gaussians from all input views are clustered with a compound hash of

* their position in disparity space ``(x/z, y/z, 1/z)`` -- a uniform grid in
  this space gives small world-space cells near the camera and large ones far
  away -- and
* their appearance (DC color, scale, opacity) quantized with
  ``appearance_resolution``,

so two splats merge only when both keys agree. Within each cluster the
attributes are combined with softmax weights computed from learned per-splat
logits (temperature-scaled). Cluster assignment runs without gradients; the
weighted sums are differentiable with respect to the attributes and logits.
"""

from __future__ import annotations

import torch

from stereosplat.rendering.gaussians import Gaussians


@torch.no_grad()
def grid_assignments_depth_adaptive(
    positions: torch.Tensor,
    harmonics: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    cell_size: float,
    appearance_resolution: float,
    min_depth_clamp: float = 0.1,
) -> tuple[torch.Tensor, int]:
    """Cluster ids from disparity-space + appearance hashing.

    Args:
        positions:  [G, 3] world-space coordinates.
        harmonics:  [G, 3, d_sh] spherical-harmonic coefficients.
        scales:     [G, 3] per-axis scales.
        opacities:  [G] opacity values.
        cell_size:  Grid resolution in disparity space (larger -> more merging).
        appearance_resolution: Bin size for color/scale/opacity quantization.
            If <= 0, only the spatial key is used.
        min_depth_clamp: Lower clamp on z before inversion.

    Returns:
        assignments: [G] long tensor with values in ``[0, K)``.
        K: number of clusters.
    """
    z = positions[:, 2].clamp(min=min_depth_clamp)
    inv_z = 1.0 / z

    disp_pos = torch.stack(
        [
            positions[:, 0] * inv_z,
            positions[:, 1] * inv_z,
            inv_z,
        ],
        dim=-1,
    )

    cells = torch.floor(disp_pos / cell_size).long()
    spatial_key = (
        cells[:, 0] * 73856093 ^ cells[:, 1] * 19349663 ^ cells[:, 2] * 83492791
    )

    if appearance_resolution <= 0:
        _, inverse = torch.unique(spatial_key, return_inverse=True)
        K = int(inverse.max().item()) + 1
        return inverse, K

    res = appearance_resolution
    color_dc = harmonics[:, :, 0]
    color_bins = torch.floor(color_dc / res).long()
    scale_bin = torch.floor(torch.log1p(scales.norm(dim=-1)) / res).long()
    opacity_bin = torch.floor(opacities / res).long()

    appearance_key = (
        color_bins[:, 0] * 314159
        ^ color_bins[:, 1] * 271828
        ^ color_bins[:, 2] * 161803
        ^ scale_bin * 141421
        ^ opacity_bin * 173205
    )

    combined = spatial_key * 1_000_000_007 + appearance_key
    _, inverse = torch.unique(combined, return_inverse=True)
    K = int(inverse.max().item()) + 1
    return inverse, K


def scatter_weighted_mean(
    src: torch.Tensor,
    weights: torch.Tensor,
    index: torch.Tensor,
    K: int,
    temperature: torch.Tensor | float = 1.0,
) -> torch.Tensor:
    """Per-cluster softmax-weighted sum ``[G, D] -> [K, D]``."""
    G, D = src.shape
    device, dtype = src.device, src.dtype

    with torch.no_grad():
        max_w = torch.full((K,), float("-inf"), device=device, dtype=dtype)
        max_w.scatter_reduce_(0, index, weights, reduce="amax")

    exp_w = torch.exp((weights - max_w[index]) / temperature)
    sum_exp = torch.zeros(K, device=device, dtype=dtype)
    sum_exp.scatter_add_(0, index, exp_w)
    normalized = exp_w / (sum_exp[index] + 1e-8)

    idx = index.unsqueeze(-1).expand(G, D)
    out = torch.zeros(K, D, device=device, dtype=dtype)
    out.scatter_add_(0, idx, src * normalized.unsqueeze(-1))
    return out


def merge_attrs_weighted(
    means: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    harmonics: torch.Tensor,
    opacities: torch.Tensor,
    assignments: torch.Tensor,
    K: int,
    weights: torch.Tensor,
    temperature: torch.Tensor | float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Merge Gaussian attributes per cluster with temperature-scaled softmax weights.

    Args:
        means: [G, 3]; scales: [G, 3]; rotations: [G, 4] (wxyz);
        harmonics: [G, 3, d_sh]; opacities: [G];
        assignments: [G] cluster ids; K: number of clusters;
        weights: [G, 1] logits; temperature: softmax temperature.
    """
    G, C, D_sh = harmonics.shape
    w = weights[:, 0]

    agg_means = scatter_weighted_mean(means, w, assignments, K, temperature)
    agg_scales = scatter_weighted_mean(scales, w, assignments, K, temperature)
    agg_rots = scatter_weighted_mean(rotations, w, assignments, K, temperature)
    agg_rots = agg_rots / (agg_rots.norm(dim=-1, keepdim=True) + 1e-8)
    agg_h = scatter_weighted_mean(
        harmonics.reshape(G, C * D_sh), w, assignments, K, temperature
    )
    agg_harmonics = agg_h.reshape(K, C, D_sh)
    agg_opacities = scatter_weighted_mean(
        opacities.unsqueeze(-1), w, assignments, K, temperature
    ).squeeze(-1)

    return agg_means, agg_scales, agg_rots, agg_harmonics, agg_opacities


def _pad(tensor: torch.Tensor, n: int) -> torch.Tensor:
    """Pad *n* zero rows along dim 0 (zero-opacity ghosts are ignored by the renderer)."""
    return torch.cat([tensor, tensor.new_zeros(n, *tensor.shape[1:])])


def aggregate_gaussians(
    gaussians: Gaussians,
    weights: torch.Tensor,
    temperature: torch.Tensor | float,
    merge_distance: float,
    appearance_resolution: float,
) -> Gaussians:
    """Merge Gaussians of each batch element independently.

    Args:
        gaussians: Per-pixel Gaussians, tensors shaped ``[B, G, ...]``.
        weights: ``[B, G, 1]`` per-splat logits.
        temperature: Softmax temperature.
        merge_distance: Disparity-space grid cell size.
        appearance_resolution: Appearance bin size.

    Returns:
        Merged Gaussians ``[B, K, ...]``; for ``B > 1`` smaller results are
        zero-padded to the largest K.
    """
    B = gaussians.means.shape[0]

    def _merge_single(
        b: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            assignments, K = grid_assignments_depth_adaptive(
                gaussians.means[b].detach(),
                gaussians.harmonics[b].detach(),
                gaussians.scales[b].detach(),
                gaussians.opacities[b].detach(),
                merge_distance,
                appearance_resolution,
            )
        return merge_attrs_weighted(
            gaussians.means[b],
            gaussians.scales[b],
            gaussians.rotations[b],
            gaussians.harmonics[b],
            gaussians.opacities[b],
            assignments,
            K,
            weights[b],
            temperature,
        )

    if B == 1:
        m, s, r, h, o = _merge_single(0)
        return Gaussians(
            means=m.unsqueeze(0),
            scales=s.unsqueeze(0),
            rotations=r.unsqueeze(0),
            harmonics=h.unsqueeze(0),
            opacities=o.unsqueeze(0),
        )

    batch_m, batch_s, batch_r, batch_h, batch_o = [], [], [], [], []
    for b in range(B):
        m, s, r, h, o = _merge_single(b)
        batch_m.append(m)
        batch_s.append(s)
        batch_r.append(r)
        batch_h.append(h)
        batch_o.append(o)

    max_k = max(t.shape[0] for t in batch_m)
    for i in range(B):
        pad = max_k - batch_m[i].shape[0]
        if pad > 0:
            batch_m[i] = _pad(batch_m[i], pad)
            batch_s[i] = _pad(batch_s[i], pad)
            batch_r[i] = _pad(batch_r[i], pad)
            batch_h[i] = _pad(batch_h[i], pad)
            batch_o[i] = _pad(batch_o[i], pad)

    return Gaussians(
        means=torch.stack(batch_m),
        scales=torch.stack(batch_s),
        rotations=torch.stack(batch_r),
        harmonics=torch.stack(batch_h),
        opacities=torch.stack(batch_o),
    )
