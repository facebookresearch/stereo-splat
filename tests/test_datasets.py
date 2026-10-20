# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import numpy as np
import pytest

from stereosplat.datasets.factory import check_dataset_not_empty, get_dataset

POSE_FILES = {
    "tartan": "poses_left.npy",
    "tartanv2": "poses_lcam_front.npy",
    "infinigen": "poses_left.npy",
    "kitti": "poses_left.npy",
    "vkitti2": "poses_left.npy",
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
