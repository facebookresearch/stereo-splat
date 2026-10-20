# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math

import numpy as np
import torch

from stereosplat.evaluation.eval_sampler import EvalSampler
from stereosplat.evaluation.evaluator import RenderingEvaluator
from stereosplat.evaluation.external_poses import (
    align_depth_scale,
    replace_poses_in_batch,
)
from stereosplat.losses.rendering_losses import ssim
from stereosplat.utils.batch import prepare_input_output_data


def test_psnr() -> None:
    ev = RenderingEvaluator(device="cpu")
    a = torch.zeros(2, 3, 8, 8)
    b = torch.full((2, 3, 8, 8), 0.1)
    assert math.isclose(ev.compute_psnr(a, b).item(), 20.0, rel_tol=1e-5)
    mask = torch.zeros(2, 1, 8, 8)
    mask[:, :, :4] = 1.0
    b2 = b.clone()
    b2[:, :, 4:] = 1.0  # outside the mask, ignored
    assert math.isclose(ev.compute_psnr(a, b2, mask).item(), 20.0, rel_tol=1e-5)


def test_depth_metrics() -> None:
    ev = RenderingEvaluator(device="cpu")
    gt = torch.full((1, 2, 4, 4), 2.0)
    pred = gt.clone()
    pred[0, 0] = 3.0  # ratio 1.5 -> outside delta1, absrel 0.5
    m = ev.compute_depth_metrics(pred, gt)
    assert math.isclose(m["absrel"], 0.25, rel_tol=1e-6)
    assert math.isclose(m["delta1"], 50.0, rel_tol=1e-6)
    assert math.isclose(m["l1"], 0.5, rel_tol=1e-6)
    gt[0, 1] = 0.0  # invalid pixels are excluded
    m = ev.compute_depth_metrics(pred, gt)
    assert math.isclose(m["delta1"], 0.0, abs_tol=1e-6)


def test_msssim_identical_images() -> None:
    ev = RenderingEvaluator(device="cpu")
    x = torch.rand(1, 3, 192, 192)
    assert math.isclose(ev.compute_msssim(x, x).item(), 1.0, rel_tol=1e-5)


def test_ssim_loss() -> None:
    x = torch.rand(2, 3, 32, 32)
    assert math.isclose(ssim(x, x).item(), 1.0, rel_tol=1e-5)
    assert ssim(x, torch.rand(2, 3, 32, 32)).item() < 0.5


def test_eval_sampler_covers_dataset_once() -> None:
    ds = list(range(10))
    shards = [list(EvalSampler(ds, num_replicas=3, rank=r)) for r in range(3)]
    assert sorted(sum(shards, [])) == ds


def test_nvs_split() -> None:
    n = 6
    data = {
        "frame_number": [[0, 1, 2]],
        "seq_name": ["s"],
        "images": torch.arange(n).float().view(1, n, 1, 1, 1).expand(1, n, 3, 2, 2),
        "original_images": torch.zeros(1, n, 3, 2, 2),
        "depths": torch.ones(1, n, 1, 2, 2),
        "intrinsics": torch.eye(3).repeat(1, n, 1, 1),
        "poses": torch.eye(4).repeat(1, n, 1, 1),
    }
    inp, out = prepare_input_output_data(data, torch.device("cpu"), 80.0, eval_nvs=True)
    assert inp["images"][0, :, 0, 0, 0].tolist() == [0.0, 2.0, 3.0, 5.0]
    assert out["images"][0, :, 0, 0, 0].tolist() == [1.0, 4.0]


def test_replace_poses(tmp_path) -> None:
    seq_dir = tmp_path / "kitti" / "seq0"
    seq_dir.mkdir(parents=True)
    c2w = np.tile(np.eye(4, dtype=np.float32), (5, 1, 1))
    c2w[:, 2, 3] = np.arange(5, dtype=np.float32)  # moving forward 1 m per frame
    np.save(seq_dir / "poses_left_aligned_scaled.npy", c2w)

    poses = torch.eye(4).repeat(1, 4, 1, 1)
    poses[0, 2:, 0, 3] = -0.5  # right camera (w2c) offset by the 0.5 m baseline
    data = {"poses": poses, "seq_name": ["seq0"], "frame_number": [[2, 4]]}
    data, n, total = replace_poses_in_batch(data, str(tmp_path), "kitti")
    assert (n, total) == (1, 1)
    out = data["poses"][0]
    assert torch.allclose(out[0], torch.eye(4))
    assert math.isclose(out[1, 2, 3].item(), -2.0)  # second frame 2 m ahead
    assert math.isclose(out[3, 0, 3].item(), -0.5)
    assert math.isclose(out[3, 2, 3].item(), -2.0)


def test_align_depth_scale() -> None:
    pred = {"depths": torch.full((1, 1, 4, 4), 2.0)}
    target = {
        "depths": torch.full((1, 1, 1, 4, 4), 4.0),
        "masks": torch.ones(1, 1, 1, 4, 4),
    }
    stats = align_depth_scale(pred, target)
    assert math.isclose(stats["depth_scale"], 2.0)
    assert torch.allclose(pred["depths"], torch.full((1, 1, 4, 4), 4.0))
