# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests of the demo pipeline up to rendering (stub backbone)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from conftest import MODEL_CONFIG, StubBackbone

from stereosplat.inference import (
    build_inference_model,
    find_stereo_sequence,
    load_poses,
    orbit_cameras,
    predict_gaussians,
    prepare_stereo_inputs,
    require_cuda,
    scene_target_depth,
    stereo_world_to_camera,
)
from stereosplat.utils.ply import load_ply, save_ply


def random_pair(
    h: int = 48, w: int = 80, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return (
        rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8),
        rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8),
    )


def test_stereo_world_to_camera() -> None:
    c2w = np.tile(np.eye(4), (2, 1, 1))
    c2w[:, 0, 3] = 5.0  # global offset is removed
    c2w[1, 2, 3] += 1.0  # second frame 1 m forward
    w2c = stereo_world_to_camera(c2w, baseline=0.3)
    centers = np.linalg.inv(w2c)[:, :3, 3]
    np.testing.assert_allclose(
        centers, [[0, 0, 0], [0, 0, 1], [0.3, 0, 0], [0.3, 0, 1]], atol=1e-12
    )
    with pytest.raises(ValueError):
        stereo_world_to_camera(c2w, baseline=0.0)


def test_prepare_inputs_resizes_and_adjusts_intrinsics() -> None:
    pair = random_pair(h=48, w=80)
    inputs = prepare_stereo_inputs(
        [pair], (60.0, 60.0, 40.0, 24.0), 0.1, img_size=(32, 64)
    )
    assert inputs.images.shape == (1, 2, 3, 32, 64)
    assert inputs.original_images.min() >= 0 and inputs.original_images.max() <= 1
    # scale = max(32/48, 64/80) = 0.8 -> 38x64, crop 3 rows
    K = inputs.intrinsics[0, 0]
    np.testing.assert_allclose(K[0].tolist(), [48.0, 0.0, 32.0], atol=1e-5)
    np.testing.assert_allclose(K[1].tolist(), [0.0, 48.0, 19.2 - 3.0], atol=1e-5)
    baseline = torch.linalg.norm(inputs.poses[0, 0, :3, 3] - inputs.poses[0, 1, :3, 3])
    assert abs(float(baseline) - 0.1) < 1e-6
    with pytest.raises(ValueError):
        prepare_stereo_inputs([pair, pair], (60.0, 60.0, 40.0, 24.0), 0.1)


def test_sequence_and_poses_io(tmp_path) -> None:
    import cv2

    for side in ("left", "right"):
        (tmp_path / side).mkdir()
        for i in range(3):
            cv2.imwrite(str(tmp_path / side / f"{i:06d}.png"), random_pair(seed=i)[0])
    pairs = find_stereo_sequence(tmp_path)
    assert [p[0].name for p in pairs] == ["000000.png", "000001.png", "000002.png"]

    poses = np.tile(np.eye(4), (3, 1, 1))
    np.savetxt(tmp_path / "poses.txt", poses[:, :3, :].reshape(3, 12))
    np.save(tmp_path / "poses.npy", poses)
    np.testing.assert_allclose(load_poses(tmp_path / "poses.txt"), poses)
    np.testing.assert_allclose(load_poses(tmp_path / "poses.npy"), poses)


def test_orbit_cameras() -> None:
    w2c = orbit_cameras(4, radius=0.5, target_depth=5.0)
    assert w2c.shape == (5, 4, 4)
    np.testing.assert_allclose(w2c[0], np.eye(4))
    c2w = np.linalg.inv(w2c)
    for pose in c2w[1:]:
        R = pose[:3, :3]
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-9)
        assert np.linalg.det(R) > 0
        center = pose[:3, 3]
        assert abs(np.linalg.norm(center[:2]) - 0.5) < 1e-9
        # optical axis points at the target
        direction = (np.array([0, 0, 5.0]) - center) / np.linalg.norm(
            [0, 0, 5.0] - center
        )
        np.testing.assert_allclose(R[:, 2], direction, atol=1e-9)


def test_pipeline_release_checkpoint_to_ply(tmp_path, stub_model) -> None:
    head = {
        k: v
        for k, v in stub_model.state_dict().items()
        if not k.startswith("fs_model.")
    }
    config = {k: v for k, v in MODEL_CONFIG.items() if k != "fs_checkpoint_path"}
    ckpt = tmp_path / "release.pth"
    torch.save({"version": 1, "config": config, "state_dict": head}, ckpt)

    torch.manual_seed(1)
    backbone = StubBackbone().eval()
    backbone.load_state_dict(stub_model.fs_model.state_dict())
    model = build_inference_model(ckpt, "unused", backbone=backbone)
    for k, v in stub_model.state_dict().items():
        assert torch.equal(model.state_dict()[k], v), k

    pairs = [random_pair(seed=0), random_pair(seed=1)]
    c2w = np.tile(np.eye(4), (2, 1, 1))
    c2w[1, 2, 3] = 0.2
    inputs = prepare_stereo_inputs(
        pairs, (60.0, 60.0, 40.0, 24.0), 0.1, left_c2w=c2w, img_size=(32, 64)
    )
    gaussians, depths = predict_gaussians(model, inputs)
    assert depths.shape == (4, 32, 64)
    assert gaussians.means.shape[0] == 1 and gaussians.means.shape[1] > 0
    assert scene_target_depth(depths) > 0

    n = save_ply(tmp_path / "out.ply", gaussians, sh_degree=3)
    back = load_ply(tmp_path / "out.ply")
    assert back.means.shape == (n, 3)
    assert np.all(back.scales > 0) and np.all(
        (back.opacities > 0) & (back.opacities < 1)
    )


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the no-CUDA error")
def test_require_cuda_message() -> None:
    with pytest.raises(RuntimeError, match="CUDA"):
        require_cuda()
