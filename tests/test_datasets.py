# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from stereosplat.datasets.factory import check_dataset_not_empty, get_dataset

POSE_FILES = {
    "tartan": "poses_left.npy",
    "tartanv2": "poses_lcam_front.npy",
    "infinigen": "poses_left.npy",
    "kitti": "poses_left.npy",
    "xrstereo": "poses_left.npy",
}


def _config(root, **kwargs) -> dict:
    cfg = {
        "dataset_path": str(root),
        "img_size": [32, 64],
        "num_views": 3,
        "stride": 1,
        "num_samples": 0,
    }
    cfg.update(kwargs)
    return cfg


@pytest.mark.parametrize("name", sorted(POSE_FILES))
def test_missing_scene_raises_with_path(tmp_path, name) -> None:
    (tmp_path / "present").mkdir()
    np.save(tmp_path / "present" / POSE_FILES[name], np.tile(np.eye(4), (9, 1, 1)))
    scenes = ["present", "absent"]
    with pytest.raises(FileNotFoundError) as err:
        get_dataset(name, _config(tmp_path), None, scene_names=scenes)
    assert str(tmp_path / "absent" / POSE_FILES[name]) in str(err.value)
    assert "allow_missing_scenes" in str(err.value)

    ds = get_dataset(
        name, _config(tmp_path, allow_missing_scenes=True), None, scene_names=scenes
    )
    assert len(ds) == 3
    assert len(ds.missing_scenes) == 1
    summary = check_dataset_not_empty(name, ds, _config(tmp_path))
    assert "1 scenes, 3 windows" in summary and "1 missing" in summary


def test_empty_dataset_raises(tmp_path) -> None:
    ds = get_dataset("tartanv2", _config(tmp_path), None, scene_names=[])
    with pytest.raises(ValueError, match="empty"):
        check_dataset_not_empty("tartanv2", ds, _config(tmp_path))
    ds = get_dataset(
        "tartanv2",
        _config(tmp_path, allow_missing_scenes=True),
        None,
        scene_names=["absent"],
    )
    with pytest.raises(ValueError, match=str(tmp_path)):
        check_dataset_not_empty("tartanv2", ds, _config(tmp_path))


def _write_xrstereo_sequence(
    seq_dir, num_frames=5, size=16, f=12.0, baseline=0.1, seed=0
) -> np.ndarray:
    """Write a synthetic XRStereo sequence; return the Blender left poses."""
    rng = np.random.default_rng(seed)
    seq_dir.mkdir(parents=True)
    c = (size - 1) / 2
    (seq_dir / "calibration.json").write_text(
        json.dumps({"f": f, "cx": c, "cy": c, "baseline_m": baseline})
    )
    poses = np.tile(np.eye(4), (num_frames, 1, 1))
    poses[:, :3, 3] = rng.normal(size=(num_frames, 3))
    angle = 0.3
    poses[:, :3, :3] = [
        [np.cos(angle), 0.0, np.sin(angle)],
        [0.0, 1.0, 0.0],
        [-np.sin(angle), 0.0, np.cos(angle)],
    ]
    np.save(seq_dir / "poses_left.npy", poses)
    for i in range(num_frames):
        disp = np.full((size, size), 2.0, dtype=np.float16)
        disp[0, :] = 0.0  # invalid (no surface)
        np.savez_compressed(
            seq_dir / f"frame_{i:04d}.npz",
            image_left=rng.integers(0, 256, (size, size, 3), dtype=np.uint8),
            image_right=rng.integers(0, 256, (size, size, 3), dtype=np.uint8),
            disparity_left=disp,
            disparity_right=disp * np.float16(2.0),
        )
    return poses


def test_xrstereo_sample(tmp_path) -> None:
    f, baseline = 12.0, 0.1
    blender_poses = _write_xrstereo_sequence(tmp_path / "seq_a", f=f, baseline=baseline)
    _write_xrstereo_sequence(tmp_path / "seq_b", seed=1)
    cfg = _config(tmp_path, img_size=[16, 16])
    ds = get_dataset(
        "xrstereo", cfg, lambda img: torch.zeros(3), scene_names=["seq_a", "seq_b"]
    )
    # 5 frames, 3 views, overlap 1 -> one window (frames 0-2) per sequence.
    assert len(ds) == 2
    assert (
        check_dataset_not_empty("xrstereo", ds, cfg) == "xrstereo: 2 scenes, 2 windows"
    )

    sample = ds[0]
    assert sample["seq_name"] == "seq_a"
    assert sample["frame_number"] == [0, 1, 2]
    assert sample["images"].shape == (6, 3)
    assert sample["original_images"].shape == (6, 3, 16, 16)
    assert sample["depths"].shape == (6, 1, 16, 16)
    assert sample["intrinsics"].shape == (6, 3, 3)

    # Depth = f * baseline / disparity; disparity 0 -> depth 0.
    depths = sample["depths"][:, 0]
    assert torch.allclose(depths[:3, 1:], torch.tensor(f * baseline / 2.0))
    assert torch.allclose(depths[3:, 1:], torch.tensor(f * baseline / 4.0))
    assert torch.all(depths[:, 0] == 0)
    assert torch.allclose(sample["intrinsics"][0, 0, 0], torch.tensor(f))

    # w2c relative to the first left camera; the right camera sits at
    # +baseline along the left camera's x axis (OpenCV axes).
    poses = sample["poses"].double()
    assert torch.allclose(poses[0], torch.eye(4, dtype=torch.float64), atol=1e-4)
    for i in range(3):
        left_to_right = poses[3 + i] @ torch.linalg.inv(poses[i])
        expected = torch.eye(4, dtype=torch.float64)
        expected[0, 3] = -baseline
        assert torch.allclose(left_to_right, expected, atol=1e-4)

    # Relative motion matches the Blender poses converted to OpenCV axes.
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    c2w = blender_poses[:3] @ flip
    rel_w2c_1 = np.linalg.inv(c2w[1]) @ c2w[0]
    assert np.allclose(poses[1].numpy(), rel_w2c_1, atol=1e-4)
