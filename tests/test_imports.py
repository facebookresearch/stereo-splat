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
    "stereosplat.datasets.tartan",
    "stereosplat.evaluation.eval_sampler",
    "stereosplat.evaluation.evaluator",
    "stereosplat.evaluation.external_poses",
    "stereosplat.losses.factory",
    "stereosplat.losses.rendering_losses",
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


def test_evaluation_does_not_import_training_code() -> None:
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py").read_text()
    assert "import train" not in src and "from train" not in src
    assert "stereosplat.losses" not in src
