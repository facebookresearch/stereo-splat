# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
KITTI Dataset for reading from local disk.

This dataset reads processed KITTI data from local disk,
supporting stereo video sequences with pose information (no depth).

Data format (individual files):
    {dataset_path}/{scene_name}/
        rgb_left/
            000000.png, 000001.png, ...
        rgb_right/
            000000.png, 000001.png, ...
        poses_left.npy
        poses_right.npy
        intrinsics_left.npy
        intrinsics_right.npy

Note: KITTI does not have dense depth, so depth is set to zeros.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import Dataset

from stereosplat.datasets.io_utils import (
    adjust_intrinsics_manual,
    create_overlapping_windows,
    get_num_frames_from_poses_individual,
    get_resize_crop_params,
    load_depth_frames_individual,
    load_intrinsics_individual,
    load_poses_individual,
    load_rgb_frames_individual,
    process_depth_or_mask,
    process_image,
)

if TYPE_CHECKING:
    from collections.abc import Callable


class KITTIDataset(Dataset):
    """
    KITTI Dataset that reads from local disk.

    Uses individual files format for efficient per-frame access:
    - RGB frames as PNG files
    - Poses and intrinsics as single NPY files per trajectory

    This format provides:
    - Per-frame access without downloading entire files
    - Better prefetching and parallel download capability
    - Lower memory overhead

    Note: KITTI does not have dense depth, so depth is set to zeros.

    Args:
        path: Base path to dataset (local directory)
        target_size: Target image size as (height, width) tuple
        transforms: Torchvision transforms to apply to images
        scene_names: List of scene names to include
        num_views: Number of views per window
        stride: Stride for frame sampling
        num_samples: Maximum number of samples (0 for all)
    """

    def __init__(
        self,
        path: str,
        target_size: tuple,
        transforms: Callable[[np.ndarray], torch.Tensor],
        scene_names: list,
        num_views: int,
        stride: int,
        num_samples: int,
    ) -> None:
        super().__init__()
        self.root = path.rstrip("/")
        self.target_size = target_size
        self.transforms = transforms
        self.stride = stride
        self.num_views = num_views
        self.num_samples = num_samples

        self.scene_names = scene_names
        self.samples: list = []
        self._create_samples()

    def _create_samples(self) -> None:
        """Create sample windows from all scenes."""
        for scene_name in self.scene_names:
            scene_path = f"{self.root}/{scene_name}"
            poses_path = f"{scene_path}/poses_left.npy"

            try:
                num_frames = get_num_frames_from_poses_individual(poses_path)
            except Exception as e:
                print(f"Warning: Could not load poses for {scene_name}: {e}")
                continue

            # Create windows with stride
            indices = np.arange(num_frames)
            strided_indices = indices[:: self.stride]
            windows = create_overlapping_windows(
                strided_indices,
                window_size=self.num_views,
                overlap=1,
            )

            for window in windows:
                self.samples.append(
                    {
                        "scene_path": scene_path,
                        "frame_indices": torch.tensor(window),
                        "seq_name": scene_name,
                    }
                )

                if self.num_samples > 0 and len(self.samples) >= self.num_samples:
                    return

    def __len__(self) -> int:
        return len(self.samples)

    def _process_side(self, sample_info: dict, side: str = "left") -> dict:
        """
        Process one side (left or right) of the stereo pair.

        Args:
            sample_info: Dictionary with scene_path and frame_indices
            side: 'left' or 'right'

        Returns:
            Dictionary with images, depths, poses, and intrinsics tensors
        """
        scene_path = sample_info["scene_path"]
        indices_tensor = sample_info["frame_indices"]
        indices = indices_tensor.tolist()

        # Build paths for individual files
        rgb_dir = f"{scene_path}/rgb_{side}"
        pose_path = f"{scene_path}/poses_{side}.npy"
        intrinsics_path = f"{scene_path}/intrinsics_{side}.npy"

        # Load RGB frames from individual PNGs
        rgb_np = load_rgb_frames_individual(rgb_dir, indices)  # (N, H, W, 3)

        # Load poses (camera-to-world)
        c2w_stack = load_poses_individual(pose_path, indices)  # (N, 4, 4)

        # Load intrinsics
        K_stack = load_intrinsics_individual(intrinsics_path, indices)  # (N, 3, 3)

        # Calculate resize/crop parameters once (all frames same size)
        h_raw, w_raw = rgb_np[0].shape[:2]
        scale, new_h, new_w, sy, sx = get_resize_crop_params(
            h_raw, w_raw, self.target_size
        )

        # Load sparse LiDAR depth if available, otherwise use zeros
        depth_dir = f"{scene_path}/depth_{side}"
        has_depth = os.path.isdir(depth_dir)
        if has_depth:
            depth_stack = load_depth_frames_individual(depth_dir, indices)  # (N, H, W)
        else:
            depth_stack = np.zeros((len(indices), h_raw, w_raw), dtype=np.float32)

        # Adjust intrinsics (supports batched (N, 3, 3))
        K_adj = adjust_intrinsics_manual(K_stack, scale, sy, sx)  # (N, 3, 3)

        processed_images: list[torch.Tensor] = []
        processed_depths: list[torch.Tensor] = []
        processed_poses: list[torch.Tensor] = []
        processed_intrinsics: list[torch.Tensor] = []
        original_images: list[torch.Tensor] = []

        for i in range(len(indices)):
            # Process RGB
            img_processed = process_image(
                rgb_np[i], scale, new_h, new_w, sy, sx, self.target_size
            )
            processed_images.append(self.transforms(img_processed))

            # Keep original normalized [0, 1] for visualization
            original_images.append(
                torch.from_numpy(img_processed.astype(np.float32) / 255.0).permute(
                    2, 0, 1
                )
            )

            # Process depth (zeros)
            d_processed = process_depth_or_mask(
                depth_stack[i], scale, new_h, new_w, sy, sx, self.target_size
            )
            processed_depths.append(torch.from_numpy(d_processed).unsqueeze(0))

            # Convert c2w to w2c
            w2c = np.linalg.inv(c2w_stack[i])
            processed_poses.append(torch.from_numpy(w2c))
            processed_intrinsics.append(torch.from_numpy(K_adj[i]))

        return {
            "images": torch.stack(processed_images),  # [N, 3, H, W]
            "original_images": torch.stack(original_images),  # [N, 3, H, W]
            "depths": torch.stack(processed_depths),  # [N, 1, H, W]
            "poses": torch.stack(processed_poses),  # [N, 4, 4]
            "intrinsics": torch.stack(processed_intrinsics),  # [N, 3, 3]
        }

    def __getitem__(self, index: int) -> dict:
        """
        Get a sample from the dataset.

        Returns:
            Dictionary containing:
            - seq_name: Scene/sequence name
            - frame_number: List of frame indices
            - images: Concatenated left+right images [2*N, 3, H, W]
            - depths: Concatenated left+right depths [2*N, 1, H, W] (zeros)
            - poses: Concatenated left+right poses [2*N, 4, 4]
            - intrinsics: Concatenated left+right intrinsics [2*N, 3, 3]
            - original_images: Concatenated original images [2*N, 3, H, W]
        """
        sample_info = self.samples[index]

        left_data = self._process_side(sample_info, side="left")
        right_data = self._process_side(sample_info, side="right")

        # Normalize poses relative to first frame of left camera
        # Poses are w2c. To normalize: new_w2c = old_w2c @ inv(first_w2c)
        first_w2c_inv = torch.linalg.inv(left_data["poses"][0])
        left_data["poses"] = left_data["poses"] @ first_w2c_inv
        right_data["poses"] = right_data["poses"] @ first_w2c_inv

        return {
            "seq_name": sample_info["seq_name"],
            "frame_number": sample_info["frame_indices"].tolist(),
            # Concatenate Left + Right -> [2*N, C, H, W]
            "images": torch.cat([left_data["images"], right_data["images"]], dim=0),
            "depths": torch.cat([left_data["depths"], right_data["depths"]], dim=0),
            "poses": torch.cat([left_data["poses"], right_data["poses"]], dim=0),
            "intrinsics": torch.cat(
                [left_data["intrinsics"], right_data["intrinsics"]], dim=0
            ),
            "original_images": torch.cat(
                [left_data["original_images"], right_data["original_images"]], dim=0
            ),
        }
