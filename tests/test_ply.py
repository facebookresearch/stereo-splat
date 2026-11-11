# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import numpy as np
import pytest
import torch
from plyfile import PlyData

from stereosplat.rendering.gaussians import Gaussians
from stereosplat.utils.ply import (
    PlyGaussians,
    gaussians_to_numpy,
    load_ply,
    ply_property_names,
    save_ply,
)


def random_gaussians(num: int = 50, sh_degree: int = 0, seed: int = 0) -> PlyGaussians:
    rng = np.random.default_rng(seed)
    rot = rng.normal(size=(num, 4)).astype(np.float32)
    return PlyGaussians(
        means=rng.normal(size=(num, 3)).astype(np.float32),
        scales=rng.uniform(1e-3, 0.5, size=(num, 3)).astype(np.float32),
        rotations=rot / np.linalg.norm(rot, axis=-1, keepdims=True),
        harmonics=rng.normal(size=(num, 3, (sh_degree + 1) ** 2)).astype(np.float32),
        opacities=rng.uniform(0.01, 0.99, size=num).astype(np.float32),
    )


def test_property_names() -> None:
    names = ply_property_names(3)
    assert names[:9] == ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    assert names[9:54] == [f"f_rest_{i}" for i in range(45)]
    assert names[54:] == ["opacity"] + [f"scale_{i}" for i in range(3)] + [
        f"rot_{i}" for i in range(4)
    ]
    assert len(ply_property_names(0)) == 17


@pytest.mark.parametrize("sh_degree", [0, 1])
def test_roundtrip(tmp_path, sh_degree: int) -> None:
    g = random_gaussians(sh_degree=sh_degree)
    path = tmp_path / "g.ply"
    assert save_ply(path, g) == 50

    ply = PlyData.read(str(path))
    vertex = ply["vertex"]
    assert [p.name for p in vertex.properties] == ply_property_names(sh_degree)
    assert all(p.val_dtype in ("f4", "float", "float32") for p in vertex.properties)

    # Stored values use the 3DGS parameterization (inverse activations).
    np.testing.assert_allclose(
        vertex["opacity"], np.log(g.opacities / (1 - g.opacities)), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(vertex["scale_0"], np.log(g.scales[:, 0]), rtol=1e-6)
    np.testing.assert_allclose(vertex["rot_0"], g.rotations[:, 0], rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(vertex["f_dc_1"], g.harmonics[:, 1, 0])
    np.testing.assert_array_equal(vertex["nx"], 0.0)
    if sh_degree == 1:
        # channel-major: f_rest_0..2 are the R coefficients, 3..5 G, 6..8 B
        np.testing.assert_allclose(vertex["f_rest_3"], g.harmonics[:, 1, 1])
        np.testing.assert_allclose(vertex["f_rest_8"], g.harmonics[:, 2, 3])

    back = load_ply(path)
    for name in ("means", "scales", "rotations", "harmonics", "opacities"):
        np.testing.assert_allclose(
            getattr(back, name), getattr(g, name), rtol=1e-5, atol=1e-6, err_msg=name
        )


def test_sh_padding(tmp_path) -> None:
    g = random_gaussians(sh_degree=0)
    path = tmp_path / "g.ply"
    save_ply(path, g, sh_degree=3)
    names = [p.name for p in PlyData.read(str(path))["vertex"].properties]
    assert names == ply_property_names(3)
    back = load_ply(path)
    assert back.harmonics.shape == (50, 3, 16)
    np.testing.assert_allclose(back.harmonics[:, :, 0], g.harmonics[:, :, 0])
    assert np.all(back.harmonics[:, :, 1:] == 0)


def test_from_batched_gaussians_drops_padding(tmp_path) -> None:
    g = random_gaussians(num=10)
    opac = torch.from_numpy(g.opacities).clone()
    opac[7:] = 0.0  # zero-opacity padding rows
    batched = Gaussians(
        means=torch.from_numpy(g.means)[None],
        scales=torch.from_numpy(g.scales)[None],
        rotations=torch.from_numpy(g.rotations)[None],
        harmonics=torch.from_numpy(g.harmonics)[None],
        opacities=opac[None],
    )
    assert gaussians_to_numpy(batched).means.shape == (7, 3)
    assert save_ply(tmp_path / "g.ply", batched) == 7
    back = load_ply(tmp_path / "g.ply")
    np.testing.assert_allclose(back.means, g.means[:7], rtol=1e-6)
