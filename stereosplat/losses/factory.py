# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from stereosplat.losses.rendering_losses import RenderingLoss


def get_loss(config: dict[str, float | str]) -> RenderingLoss:
    name = config["name"]
    if name == "l1":
        return RenderingLoss(config)
    else:
        raise ValueError(f"Unknown loss: {name}")
