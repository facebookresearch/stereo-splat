# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Read and write Gaussians in the standard 3D Gaussian Splatting ``.ply`` layout.

The layout is the one written by the original 3DGS implementation and read by
common viewers: one ``vertex`` element with the float32 properties

    x, y, z, nx, ny, nz,
    f_dc_0..2, f_rest_0..(3 * (d_sh - 1) - 1),
    opacity, scale_0..2, rot_0..3

where ``opacity`` is stored as a logit, ``scale_*`` as the natural log of the
standard deviations, ``rot_*`` as a (w, x, y, z) quaternion, ``f_dc_*`` are the
degree-0 SH coefficients (RGB) and ``f_rest_*`` the higher-order coefficients
ordered channel-major (all coefficients of R, then G, then B). Normals are
written as zeros.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
from plyfile import PlyData, PlyElement

from stereosplat.rendering.gaussians import Gaussians

_EPS = 1e-6


@dataclass
class PlyGaussians:
    """Activated Gaussian parameters of a single scene (numpy, float32).

    Attributes:
        means: [G, 3] centers.
        scales: [G, 3] standard deviations (> 0).
        rotations: [G, 4] unit quaternions (w, x, y, z).
        harmonics: [G, 3, d_sh] SH coefficients per color channel.
        opacities: [G] opacities in (0, 1).
    """

    means: np.ndarray
    scales: np.ndarray
    rotations: np.ndarray
    harmonics: np.ndarray
    opacities: np.ndarray


def ply_property_names(sh_degree: int) -> list[str]:
    """Property names of the ``vertex`` element for a given SH degree."""
    num_rest = 3 * ((sh_degree + 1) ** 2 - 1)
    return (
        ["x", "y", "z", "nx", "ny", "nz"]
        + [f"f_dc_{i}" for i in range(3)]
        + [f"f_rest_{i}" for i in range(num_rest)]
        + ["opacity"]
        + [f"scale_{i}" for i in range(3)]
        + [f"rot_{i}" for i in range(4)]
    )


def gaussians_to_numpy(
    gaussians: Gaussians, batch_index: int = 0, min_opacity: float = 0.0
) -> PlyGaussians:
    """Extract one scene of a batched :class:`Gaussians` as numpy arrays.

    Gaussians with opacity ``<= min_opacity`` are dropped; this removes the
    zero-opacity padding added when scenes of a batch have different sizes.
    """

    def get(t: torch.Tensor) -> np.ndarray:
        return t[batch_index].detach().float().cpu().numpy()

    opacities = get(gaussians.opacities)
    if opacities.ndim > 1:  # opacity SH: keep the DC term
        opacities = opacities.reshape(opacities.shape[0], -1)[:, 0]
    keep = opacities > min_opacity
    return PlyGaussians(
        means=get(gaussians.means)[keep],
        scales=get(gaussians.scales)[keep],
        rotations=get(gaussians.rotations)[keep],
        harmonics=get(gaussians.harmonics)[keep],
        opacities=opacities[keep],
    )


def save_ply(
    path: str | os.PathLike[str],
    gaussians: PlyGaussians | Gaussians,
    sh_degree: int | None = None,
) -> int:
    """Write Gaussians to ``path`` in the 3DGS ``.ply`` layout.

    Args:
        path: Output file.
        gaussians: A :class:`PlyGaussians` or a batched :class:`Gaussians`
            (the first scene is written).
        sh_degree: SH degree of the written file. Defaults to the degree of the
            input. A higher degree zero-pads the higher-order coefficients
            (some viewers expect degree 3); a lower degree truncates them.

    Returns:
        The number of Gaussians written.
    """
    if isinstance(gaussians, Gaussians):
        gaussians = gaussians_to_numpy(gaussians)
    g = gaussians
    num = g.means.shape[0]
    d_sh_in = g.harmonics.shape[-1]
    sh_degree_in = int(round(np.sqrt(d_sh_in))) - 1
    if (sh_degree_in + 1) ** 2 != d_sh_in:
        raise ValueError(f"Invalid number of SH coefficients: {d_sh_in}")
    if sh_degree is None:
        sh_degree = sh_degree_in
    d_sh = (sh_degree + 1) ** 2

    harmonics = np.zeros((num, 3, d_sh), dtype=np.float32)
    n = min(d_sh, d_sh_in)
    harmonics[:, :, :n] = g.harmonics[:, :, :n]

    opacities = np.clip(g.opacities.astype(np.float64), _EPS, 1.0 - _EPS)
    rotations = g.rotations / np.maximum(
        np.linalg.norm(g.rotations, axis=-1, keepdims=True), 1e-12
    )
    columns = [
        g.means,
        np.zeros((num, 3), dtype=np.float32),
        harmonics[:, :, 0],
        harmonics[:, :, 1:].reshape(num, -1),
        np.log(opacities / (1.0 - opacities))[:, None],
        np.log(np.maximum(g.scales, 1e-12)),
        rotations,
    ]
    data = np.concatenate([np.asarray(c, dtype=np.float32) for c in columns], axis=1)

    names = ply_property_names(sh_degree)
    assert data.shape[1] == len(names)
    vertices = np.empty(num, dtype=[(name, "f4") for name in names])
    for i, name in enumerate(names):
        vertices[name] = data[:, i]

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    PlyData([PlyElement.describe(vertices, "vertex")]).write(str(path))
    return num


def load_ply(path: str | os.PathLike[str]) -> PlyGaussians:
    """Read a 3DGS ``.ply`` file and apply the activations (inverse of :func:`save_ply`)."""
    vertex = PlyData.read(str(path))["vertex"]
    names = [p.name for p in vertex.properties]

    def stack(keys: list[str]) -> np.ndarray:
        return np.stack([np.asarray(vertex[k], dtype=np.float32) for k in keys], -1)

    num_rest = len([n for n in names if n.startswith("f_rest_")])
    if num_rest % 3:
        raise ValueError(f"{path}: number of f_rest properties not divisible by 3")
    num = len(vertex.data)
    dc = stack([f"f_dc_{i}" for i in range(3)])
    if num_rest:
        rest = stack([f"f_rest_{i}" for i in range(num_rest)])
        rest = rest.reshape(num, 3, num_rest // 3)
    else:
        rest = np.zeros((num, 3, 0), dtype=np.float32)
    harmonics = np.concatenate([dc[:, :, None], rest], axis=-1)

    logit = np.asarray(vertex["opacity"], dtype=np.float32)
    return PlyGaussians(
        means=stack(["x", "y", "z"]),
        scales=np.exp(stack([f"scale_{i}" for i in range(3)])),
        rotations=stack([f"rot_{i}" for i in range(4)]),
        harmonics=harmonics,
        opacities=1.0 / (1.0 + np.exp(-logit)),
    )
