# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Dataset factory for creating dataset instances.

Supports TartanAir (v1, training), TartanAir V2, Infinigen, KITTI and
Virtual KITTI 2, all read from the processed per-frame layout on local disk.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from stereosplat.datasets.infinigen import InfinigenDataset
from stereosplat.datasets.kitti import KITTIDataset
from stereosplat.datasets.tartan import TartanAirDataset
from stereosplat.datasets.tartanv2 import TartanAirV2Dataset
from stereosplat.datasets.vkitti2 import VKITTI2Dataset

# Repository root (the directory that contains ``data/splits``).
REPO_ROOT: Path = Path(__file__).resolve().parents[2]


def load_split_file(split_file_path: str) -> list[str]:
    """
    Load a split file containing scene names (one per line).

    Relative paths are resolved against the current working directory first
    and then against the repository root, so ``data/splits/tartanv2/test.txt``
    works from anywhere.

    Args:
        split_file_path: Absolute path, or path relative to cwd / repo root.

    Returns:
        Sorted list of scene names from the split file.
    """
    candidates = [Path(split_file_path)]
    if not os.path.isabs(split_file_path):
        candidates.append(REPO_ROOT / split_file_path)
    for path in candidates:
        if path.is_file():
            with open(path) as f:
                return sorted([line.strip() for line in f.readlines() if line.strip()])

    raise FileNotFoundError(
        f"Split file not found: {split_file_path} (also tried {candidates[-1]})."
    )


def get_dataset(
    dataset_name: str,
    dataset_config: dict[str, Any],
    transforms: Callable,
    scene_names: list[str] | None = None,
) -> (
    TartanAirDataset
    | TartanAirV2Dataset
    | InfinigenDataset
    | KITTIDataset
    | VKITTI2Dataset
):
    """
    Factory function to create dataset instances.

    Args:
        dataset_name: Name of the dataset to create
            ("tartan", "tartanv2", "infinigen", "kitti", "vkitti2")
        dataset_config: Configuration dictionary containing:
            - dataset_path: Local path to the processed dataset
            - split_file_path: Path to file listing scene names (one per line)
            - img_size: Target image size as (height, width) tuple
            - num_views: Number of views per sample
            - stride: Frame sampling stride
            - num_samples: Maximum number of samples (0 for all)
            - allow_missing_scenes: Skip (warn about) split scenes that are
              missing on disk instead of raising (default False)
        transforms: Torchvision transforms to apply to images
        scene_names: Optional list of scene names. If not provided,
            reads from split_file_path.

    Returns:
        Dataset instance

    Raises:
        ValueError: If dataset_name is not supported

    Example config:
        dataset_config = {
            "dataset_path": "/data/tartanair_v2_processed",
            "split_file_path": "data/splits/tartanv2/test.txt",
            "img_size": (480, 640),
            "num_views": 4,
            "stride": 1,
            "num_samples": 0,  # 0 means all samples
        }
    """
    # Load scene names from split file if not provided
    if scene_names is None or len(scene_names) == 0:
        split_file_path = dataset_config.get("split_file_path", "")
        if split_file_path:
            scene_names = load_split_file(split_file_path)
        else:
            scene_names = []

    dataset_classes = {
        "tartan": TartanAirDataset,
        "tartanv2": TartanAirV2Dataset,
        "infinigen": InfinigenDataset,
        "kitti": KITTIDataset,
        "vkitti2": VKITTI2Dataset,
    }

    if dataset_name not in dataset_classes:
        raise ValueError(
            f"Dataset '{dataset_name}' not supported. "
            f"Currently supported: {list(dataset_classes.keys())}"
        )

    return dataset_classes[dataset_name](
        path=dataset_config["dataset_path"],
        target_size=tuple(dataset_config["img_size"]),
        transforms=transforms,
        scene_names=scene_names,
        num_views=dataset_config.get("num_views", 2),
        stride=dataset_config.get("stride", 1),
        num_samples=dataset_config.get("num_samples", 0),
        allow_missing_scenes=bool(dataset_config.get("allow_missing_scenes", False)),
    )


def check_dataset_not_empty(
    name: str, dataset: Dataset, dataset_config: dict[str, Any]
) -> str:
    """Raise if ``dataset`` has no samples; return a one-line size summary.

    The summary lists the number of scenes that were loaded and the number of
    windows (samples), e.g. for logging at startup.
    """
    scenes = getattr(dataset, "scene_names", None) or []
    missing = getattr(dataset, "missing_scenes", None) or []
    if len(dataset) == 0:
        raise ValueError(
            f"Dataset '{name}' is empty: 0 windows from {len(scenes)} split "
            f"scene(s) ({len(missing)} missing) at "
            f"dataset_path={dataset_config.get('dataset_path')!r}, "
            f"split_file_path={dataset_config.get('split_file_path')!r}, "
            f"num_views={dataset_config.get('num_views')}, "
            f"stride={dataset_config.get('stride', 1)}. Check DATA_ROOT and the "
            "dataset config."
        )
    samples = getattr(dataset, "samples", None) or []
    num_scenes = len({sample["seq_name"] for sample in samples})
    return f"{name}: {num_scenes} scenes, {len(dataset)} windows" + (
        f" ({len(missing)} missing scenes skipped)" if missing else ""
    )
