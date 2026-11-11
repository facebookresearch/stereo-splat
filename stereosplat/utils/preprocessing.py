# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Input image preprocessing for StereoSplat."""

from __future__ import annotations

from torchvision import transforms

# Dataset-specific pixel normalization applied to model input images
# (see get_transforms). Kept as named constants so backbones that need the raw
# 0-255 space (e.g. IGEV++) can invert it.
IMAGE_NORM_MEAN: list[float] = [0.34721234, 0.36705238, 0.36066107]
IMAGE_NORM_STD: list[float] = [0.30737526, 0.31515116, 0.32020183]


def get_transforms() -> transforms.Compose:
    """
    Get the standard image transforms applied to model input images.

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
