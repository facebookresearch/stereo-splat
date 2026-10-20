# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Training utilities for StereoSplat.

This module provides utility functions for training including:
- Image transforms and preprocessing
- Output path setup
- Model freezing utilities
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from torch import nn
from torchvision import transforms

# Dataset-specific pixel normalization applied to model input images
# (see get_transforms). Kept as named constants so backbones that need the raw
# 0-255 space (e.g. IGEV++) can invert it.
IMAGE_NORM_MEAN: List[float] = [0.34721234, 0.36705238, 0.36066107]
IMAGE_NORM_STD: List[float] = [0.30737526, 0.31515116, 0.32020183]


def setup_output_paths(output_path: str, folder_names: List[str]) -> None:
    """
    Create output directories for training artifacts.

    Args:
        output_path: Base output directory path
        folder_names: List of subdirectory names to create
    """
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    for folder_name in folder_names:
        (output / folder_name).mkdir(exist_ok=True)


def get_transforms() -> transforms.Compose:
    """
    Get the standard image transforms for training.

    Returns:
        Composed transforms that convert to tensor and normalize
        using dataset-specific mean and std values.
    """
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_NORM_MEAN, std=IMAGE_NORM_STD),
        ]
    )


def freeze_model(model: nn.Module) -> nn.Module:
    """
    Freeze all parameters in a model for inference.

    Args:
        model: PyTorch model to freeze

    Returns:
        The frozen model in eval mode
    """
    model = model.eval()
    for p in model.parameters():
        p.requires_grad = False
    for p in model.buffers():
        p.requires_grad = False
    return model
