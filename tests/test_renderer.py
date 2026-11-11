# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pytest
import torch

from stereosplat.rendering.gaussians import Gaussians
from stereosplat.rendering.renderer import run_renderer_in_chunk_w_trj_mode

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="gsplat rasterization needs CUDA"
)


def test_render_single_gaussian() -> None:
    device = torch.device("cuda")
    h, w = 32, 48
    gaussians = Gaussians(
        means=torch.tensor([[[0.0, 0.0, 2.0]]], device=device),
        scales=torch.full((1, 1, 3), 0.2, device=device),
        rotations=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], device=device),
        harmonics=torch.full((1, 1, 3, 1), 1.0, device=device),
        opacities=torch.tensor([[0.99]], device=device),
    )
    K = torch.tensor(
        [[40.0, 0.0, w / 2], [0.0, 40.0, h / 2], [0.0, 0.0, 1.0]], device=device
    )
    out = run_renderer_in_chunk_w_trj_mode(
        gaussians=gaussians,
        extrinsics=torch.eye(4, device=device)[None, None],
        intrinsics=K[None, None],
        image_shape=(h, w),
    )
    assert out["colors"].shape == (1, 1, 3, h, w)
    assert out["alphas"][0, 0, h // 2, w // 2] > 0.5
    center_depth = (
        out["depths"][0, 0, h // 2, w // 2] / out["alphas"][0, 0, h // 2, w // 2]
    )
    assert abs(center_depth.item() - 2.0) < 0.1
