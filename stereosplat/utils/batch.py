# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Batch preparation for evaluation."""

from __future__ import annotations

from typing import Any

import torch


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack tensors; keep non-tensor fields (strings, frame lists) as lists."""
    elem = batch[0]
    result: dict[str, Any] = {}
    for key in elem:
        values = [d[key] for d in batch]
        if isinstance(values[0], torch.Tensor):
            result[key] = torch.stack(values)
        else:
            result[key] = values
    return result


def prepare_model_inputs(
    data: dict[str, Any],
    device: torch.device,
    max_depth: float = 30.0,
) -> dict[str, Any]:
    """Move a batch to ``device`` and compute depth-validity masks.

    Depths are clamped to ``max_depth``; ``masks`` ([B, N, 1, H, W]) is 1.0
    where the raw depth is in ``(0, max_depth)``.
    """
    return {
        "frame_number": data["frame_number"],
        "seq_name": data["seq_name"],
        "images": data["images"].to(device, non_blocking=True),
        "original_images": data["original_images"].to(device, non_blocking=True),
        "depths": data["depths"].clamp(max=max_depth).to(device, non_blocking=True),
        "intrinsics": data["intrinsics"].to(device, non_blocking=True),
        "poses": data["poses"].to(device, non_blocking=True),
        "masks": ((data["depths"] > 0) & (data["depths"] < max_depth))
        .to(device, non_blocking=True)
        .float(),
    }


def prepare_input_output_data(
    data: dict[str, Any],
    device: torch.device,
    max_depth: float = 30.0,
    eval_nvs: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a batch into model inputs and rendering targets.

    In-view synthesis (``eval_nvs=False``) uses all views as both input and
    target. For novel-view synthesis the stereo pairs are split: even pairs
    are context, odd pairs are held-out targets (with 2 pairs: pair 0 is the
    context and pair 1 the target).

    Views are ordered ``[L0, L1, ..., L_{P-1}, R0, R1, ..., R_{P-1}]``.
    """
    all_data = prepare_model_inputs(data, device, max_depth)

    if not eval_nvs:
        return all_data, all_data

    num_total = all_data["images"].shape[1]
    num_pairs = num_total // 2

    if num_pairs <= 1:
        return all_data, all_data

    if num_pairs == 2:
        input_pair_idx = [0]
        output_pair_idx = [1]
    else:
        input_pair_idx = list(range(0, num_pairs, 2))
        output_pair_idx = list(range(1, num_pairs, 2))

    input_view_idx = list(input_pair_idx) + [i + num_pairs for i in input_pair_idx]
    output_view_idx = list(output_pair_idx) + [i + num_pairs for i in output_pair_idx]

    input_data: dict[str, Any] = {}
    output_data: dict[str, Any] = {}
    for k, v in all_data.items():
        if isinstance(v, torch.Tensor) and v.dim() > 1:
            input_data[k] = v[:, input_view_idx]
            output_data[k] = v[:, output_view_idx]
        else:
            input_data[k] = v
            output_data[k] = v

    return input_data, output_data
