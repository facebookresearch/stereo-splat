# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import importlib

import pytest

MODULES = [
    "stereosplat",
    "stereosplat.backbones.foundation_stereo",
    "stereosplat.datasets.factory",
    "stereosplat.datasets.io_utils",
    "stereosplat.evaluation.eval_sampler",
    "stereosplat.evaluation.evaluator",
    "stereosplat.evaluation.external_poses",
    "stereosplat.models.aggregation",
    "stereosplat.models.build",
    "stereosplat.models.cross_view_attention",
    "stereosplat.models.gaussian_adapter",
    "stereosplat.models.layers",
    "stereosplat.models.stereosplat",
    "stereosplat.rendering.gaussians",
    "stereosplat.rendering.renderer",
    "stereosplat.utils.batch",
    "stereosplat.utils.distributed",
    "stereosplat.utils.geometry",
    "stereosplat.utils.preprocessing",
    "stereosplat.utils.sh",
    "stereosplat.utils.visualization",
]


@pytest.mark.parametrize("name", MODULES)
def test_import(name: str) -> None:
    importlib.import_module(name)
