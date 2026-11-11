# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Evaluation with externally estimated camera poses (e.g. MAC-VO).

Expected layout (one file per sequence)::

    <poses_dir>/<dataset_name>/<seq_name>/<pose_type>.npy

Each file stores an ``(N_frames, 4, 4)`` float array of camera-to-world poses
of the LEFT camera, already aligned to the ground-truth trajectory (Sim3 or
SE3). Right-camera poses are derived from the ground-truth stereo calibration.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def load_external_left_poses(
    poses_dir: str,
    dataset_name: str,
    seq_name: str,
    frame_indices: list[int],
    pose_type: str = "poses_left_aligned_scaled",
) -> torch.Tensor:
    """Load external left-camera poses for one frame window.

    The poses are re-expressed relative to the first frame of the window (the
    dataset convention) and converted to world-to-camera.

    Returns:
        ``[N, 4, 4]`` float32 tensor of left-camera w2c poses.
    """
    left_path = f"{poses_dir}/{dataset_name}/{seq_name}/{pose_type}.npy"
    left_c2w_all = np.load(left_path)

    max_idx = max(frame_indices) if frame_indices else 0
    if max_idx >= len(left_c2w_all):
        raise IndexError(
            f"Frame index {max_idx} out of range for {seq_name} "
            f"({len(left_c2w_all)} poses in {left_path})."
        )

    left_c2w = left_c2w_all[frame_indices]
    left_c2w = np.linalg.inv(left_c2w[0]) @ left_c2w
    left_w2c = np.linalg.inv(left_c2w).astype(np.float32)
    return torch.from_numpy(left_w2c)


def _frames_of(frame_numbers: Any, b: int) -> list[int]:
    if isinstance(frame_numbers, torch.Tensor):
        return frame_numbers[b].tolist()
    if isinstance(frame_numbers[0], torch.Tensor):
        return [int(t[b]) for t in frame_numbers]
    if isinstance(frame_numbers[0], (list, tuple)):
        return list(frame_numbers[b])
    return list(frame_numbers)


def replace_poses_in_batch(
    data: dict[str, Any],
    poses_dir: str,
    dataset_name: str,
    pose_type: str = "poses_left_aligned_scaled",
) -> tuple[dict[str, Any], int, int]:
    """Replace ground-truth poses in a batch with external estimates.

    Right-camera poses are ``T_stereo @ ext_left_w2c`` with the fixed rig
    transform ``T_stereo = gt_right_w2c[0] @ gt_left_c2w[0]``, which equals
    ``gt_right_w2c[0]`` because the dataset normalizes ``gt_left_w2c[0] = I``.

    Returns:
        ``(data, num_replaced, num_total)``.
    """
    batch_poses = data["poses"]  # [B, 2N, 4, 4]
    B = batch_poses.shape[0]
    num_views = batch_poses.shape[1] // 2
    seq_names = data["seq_name"]

    for b in range(B):
        seq = seq_names[b] if isinstance(seq_names, (list, tuple)) else seq_names
        frames = _frames_of(data["frame_number"], b)
        T_stereo = batch_poses[b, num_views].clone()
        left_w2c = load_external_left_poses(
            poses_dir, dataset_name, seq, frames, pose_type
        )
        if left_w2c.shape[0] != num_views:
            raise ValueError(
                f"Pose count mismatch for {seq}: expected {num_views}, got "
                f"{left_w2c.shape[0]} (frames={frames})"
            )
        batch_poses[b, :num_views] = left_w2c
        batch_poses[b, num_views:] = T_stereo.unsqueeze(0) @ left_w2c

    data["poses"] = batch_poses
    return data, B, B


def align_depth_scale(
    renderer_output: dict[str, Any],
    output_data: dict[str, Any],
) -> dict[str, float]:
    """Median-depth scale alignment of the rendered depth (evaluation-only).

    Scales ``renderer_output["depths"]`` in place by
    ``median(gt) / median(pred)`` over the valid and visible pixels of the
    whole batch (per window when ``batch_size == 1``).
    """
    metrics_scale = 1.0
    scale_computed = 0.0

    pred_d = renderer_output["depths"]
    gt_d = output_data["depths"][:, :, 0]
    valid = (output_data["masks"].view_as(pred_d) > 0) & (gt_d > 1e-8)
    if valid.sum() > 0:
        scale = torch.median(gt_d[valid]) / torch.median(pred_d[valid].clamp(min=1e-8))
        renderer_output["depths"] = pred_d * scale
        metrics_scale = float(scale.item())
        scale_computed = 1.0

    return {"depth_scale": metrics_scale, "depth_scale_computed_frac": scale_computed}
