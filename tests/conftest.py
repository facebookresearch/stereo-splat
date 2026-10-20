# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pytest
import torch
from torch import nn


class StubBackbone(nn.Module):
    """Stand-in for FoundationStereo with the same output contract.

    ``forward(image1, image2, return_features=True)`` returns a positive
    disparity ``[B, 1, H, W]`` and features ``stem`` (32 ch, 1/2), ``ctx0``,
    ``ctx1``, ``ctx2`` (128 ch at 1/4, 1/8, 1/16) for left and right views
    concatenated along the batch dimension.
    """

    def __init__(self) -> None:
        super().__init__()
        self.disp = nn.Conv2d(6, 1, 3, padding=1)
        self.stem = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.ctx0 = nn.Conv2d(3, 128, 3, stride=4, padding=1)
        self.ctx1 = nn.Conv2d(3, 128, 3, stride=8, padding=1)
        self.ctx2 = nn.Conv2d(3, 128, 3, stride=16, padding=1)

    def forward(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
        iters: int = 12,
        return_features: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        disp = 5.0 + nn.functional.softplus(self.disp(torch.cat([image1, image2], 1)))
        if not return_features:
            return disp, None
        both = torch.cat([image1, image2], 0)
        feats = {
            "stem": self.stem(both),
            "ctx0": self.ctx0(both),
            "ctx1": self.ctx1(both),
            "ctx2": self.ctx2(both),
        }
        return disp, feats


def make_window(
    batch: int = 1, num_pairs: int = 2, h: int = 32, w: int = 64, seed: int = 0
) -> dict[str, torch.Tensor]:
    """Random normalized images with a forward-moving 10 cm-baseline stereo rig."""
    g = torch.Generator().manual_seed(seed)
    n = 2 * num_pairs
    images = torch.randn(batch, n, 3, h, w, generator=g)
    poses = torch.eye(4).repeat(batch, n, 1, 1)
    for p in range(num_pairs):
        poses[:, p, 2, 3] = -0.05 * p
        poses[:, num_pairs + p, 0, 3] = -0.1
        poses[:, num_pairs + p, 2, 3] = -0.05 * p
    K = torch.tensor([[0.8 * w, 0.0, w / 2], [0.0, 0.8 * w, h / 2], [0.0, 0.0, 1.0]])
    return {
        "images": images,
        "poses": poses,
        "intrinsics": K.repeat(batch, n, 1, 1),
    }


MODEL_CONFIG = {
    "name": "stereosplat",
    "fs_checkpoint_path": "unused",
    "sh_degree": 0,
    "gaussian_scale_min": 1.0e-5,
    "gaussian_scale_max": 80.0,
    "merge_distance": 0.003,
    "appearance_resolution": 0.2,
    "weight_temperature_init": 1.0,
    "cross_view_num_layers": 4,
    "max_depth": 80.0,
}


@pytest.fixture
def stub_model():
    from stereosplat.models.build import build_model

    torch.manual_seed(0)
    model = build_model(MODEL_CONFIG, backbone=StubBackbone().eval())
    return model.eval()
