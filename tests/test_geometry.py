# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math

import torch

from stereosplat.models.aggregation import grid_assignments_depth_adaptive
from stereosplat.models.gaussian_adapter import GaussianAdapter
from stereosplat.utils.geometry import (
    affine_inverse,
    as_homogeneous,
    get_fov,
    mat_to_quat,
    quat_to_mat,
    sample_image_grid,
)


def _random_rigid(n: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(n, 4, generator=g)
    q = q / q.norm(dim=-1, keepdim=True)
    T = torch.eye(4).repeat(n, 1, 1)
    T[:, :3, :3] = quat_to_mat(q)
    T[:, :3, 3] = torch.randn(n, 3, generator=g)
    return T


def test_affine_inverse() -> None:
    T = _random_rigid(8)
    eye = torch.eye(4).expand(8, 4, 4)
    assert torch.allclose(affine_inverse(T) @ T, eye, atol=1e-5)
    assert torch.allclose(affine_inverse(T), torch.linalg.inv(T), atol=1e-5)


def test_as_homogeneous() -> None:
    T = _random_rigid(3)
    assert torch.equal(as_homogeneous(T[:, :3]), T)


def test_quaternion_roundtrip() -> None:
    R = _random_rigid(16)[:, :3, :3]
    R2 = quat_to_mat(mat_to_quat(R))
    assert torch.allclose(R, R2, atol=1e-5)


def test_sample_image_grid() -> None:
    xy, ij = sample_image_grid((4, 8))
    assert xy.shape == (4, 8, 2) and ij.shape == (4, 8, 2)
    assert torch.allclose(xy[0, 0], torch.tensor([0.5 / 8, 0.5 / 4]))
    assert torch.allclose(xy[3, 7], torch.tensor([7.5 / 8, 3.5 / 4]))


def test_get_fov() -> None:
    f = 0.5  # normalized focal length -> 90 degree field of view
    K = torch.tensor([[[f, 0.0, 0.5], [0.0, f, 0.5], [0.0, 0.0, 1.0]]])
    fov = get_fov(K)
    assert torch.allclose(fov, torch.full((1, 2), math.pi / 2), atol=1e-5)


def test_gaussian_adapter_unprojects_pixel_centers() -> None:
    h, w = 4, 6
    adapter = GaussianAdapter(sh_degree=0)
    K = torch.tensor([[5.0, 0.0, 3.0], [0.0, 5.0, 2.0], [0.0, 0.0, 1.0]])
    w2c = torch.eye(4)
    w2c[:3, 3] = torch.tensor([0.1, -0.2, 0.3])
    depth = torch.full((1, 1, 1, h, w), 2.0)
    raw = torch.zeros(1, 1, h, w, 1 + 3 + 4 + 3)
    raw[..., 4 + 3] = 1.0  # identity quaternion (xyzw)
    g = adapter(
        extrinsics=w2c[None, None],
        intrinsics=K[None, None],
        depths=depth,
        raw_gaussians=raw,
    )
    assert g.means.shape == (1, h * w, 3)
    # Project the world-space means back into the camera.
    pts = g.means[0] @ w2c[:3, :3].T + w2c[:3, 3]
    assert torch.allclose(pts[:, 2], torch.full((h * w,), 2.0), atol=1e-5)
    uv = pts @ K.T
    uv = uv[:, :2] / uv[:, 2:]
    ys, xs = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    expected = torch.stack([xs.flatten() + 0.5, ys.flatten() + 0.5], dim=-1).float()
    assert torch.allclose(uv, expected, atol=1e-4)
    assert torch.allclose(g.opacities, torch.full_like(g.opacities, 0.5))
    assert torch.allclose(g.rotations[0, :, 0].abs(), torch.ones(h * w), atol=1e-5)


def test_depth_adaptive_grid_merges_close_points_only() -> None:
    pos = torch.tensor(
        [[0.0, 0.0, 10.0], [0.0001, 0.0, 10.0001], [1.0, 0.0, 2.0]], dtype=torch.float32
    )
    harmonics = torch.zeros(3, 3, 1)
    scales = torch.full((3, 3), 0.01)
    opac = torch.full((3,), 0.5)
    assign, K = grid_assignments_depth_adaptive(
        pos, harmonics, scales, opac, 0.003, 0.2
    )
    assert K == 2
    assert assign[0] == assign[1] != assign[2]
