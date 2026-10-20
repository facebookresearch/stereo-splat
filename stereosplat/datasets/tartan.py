# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
TartanAir Dataset for reading from local disk.

This dataset reads processed TartanAir data from local disk,
supporting stereo video sequences with depth and pose information.

Data format (individual files):
    {dataset_path}/{scene_name}/
        rgb_left/
            000000.png, 000001.png, ...
        rgb_right/
            000000.png, 000001.png, ...
        depth_left/
            000000.npy, 000001.npy, ...
        depth_right/
            000000.npy, 000001.npy, ...
        poses_left.npy
        poses_right.npy
        intrinsics_left.npy
        intrinsics_right.npy
"""

from __future__ import annotations

from collections.abc import Callable

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
    report_missing_scenes,
)


class TartanAirDataset(Dataset):
    """
    TartanAir Dataset that reads from local disk.

    Uses individual files format for efficient per-frame access:
    - RGB frames as PNG files
    - Depth maps as NPY files
    - Poses and intrinsics as single NPY files per trajectory

    This format provides:
    - Per-frame access without downloading entire files
    - Better prefetching and parallel download capability
    - Lower memory overhead

    Args:
        path: Base path to dataset (local directory)
        target_size: Target image size as (height, width) tuple
        transforms: Torchvision transforms to apply to images
        scene_names: List of scene names to include
        num_views: Number of views per window
        stride: Stride for frame sampling
        num_samples: Maximum number of samples (0 for all)
        allow_missing_scenes: Skip split scenes whose poses cannot be loaded
            (with a warning) instead of raising ``FileNotFoundError``.
    """

    def __init__(
        self,
        path: str,
        target_size: tuple,
        transforms: Callable[..., torch.Tensor],
        scene_names: list,
        num_views: int,
        stride: int,
        num_samples: int,
        allow_missing_scenes: bool = False,
    ) -> None:
        super().__init__()
        self.root = path.rstrip("/")
        self.target_size = target_size
        self.transforms = transforms
        self.stride = stride
        self.num_views = num_views
        self.num_samples = num_samples

        # TartanAir Intrinsics (original resolution: 640x480)
        self.base_K = np.array(
            [
                [320.0, 0.0, 320.0],
                [0.0, 320.0, 240.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        self.scene_names = scene_names
        self.samples = []
        self.missing_scenes: list[str] = []
        self._create_samples()
        report_missing_scenes(
            type(self).__name__, self.root, self.missing_scenes, allow_missing_scenes
        )

    def _create_samples(self) -> None:
        """Create sample windows from all scenes."""
        for scene_name in self.scene_names:
            scene_path = f"{self.root}/{scene_name}"
            poses_path = f"{scene_path}/poses_left.npy"

            try:
                num_frames = get_num_frames_from_poses_individual(poses_path)
            except Exception as e:
                self.missing_scenes.append(f"{scene_name}: {poses_path} ({e})")
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
        depth_dir = f"{scene_path}/depth_{side}"
        pose_path = f"{scene_path}/poses_{side}.npy"
        intrinsics_path = f"{scene_path}/intrinsics_{side}.npy"

        # Load RGB frames from individual PNGs
        rgb_np = load_rgb_frames_individual(rgb_dir, indices)  # (N, H, W, 3)

        # Load depth from individual NPYs
        depth_stack = load_depth_frames_individual(depth_dir, indices)  # (N, H, W)

        # Load poses (camera-to-world)
        c2w_stack = load_poses_individual(pose_path, indices)  # (N, 4, 4)

        # Load intrinsics (or use base_K if loading fails)
        try:
            K_stack = load_intrinsics_individual(intrinsics_path, indices)  # (N, 3, 3)
        except Exception:
            K_stack = np.broadcast_to(
                self.base_K, (len(indices), 3, 3)
            ).copy()  # (N, 3, 3)

        # Calculate resize/crop parameters once (all frames same size)
        h_raw, w_raw = rgb_np[0].shape[:2]
        scale, new_h, new_w, sy, sx = get_resize_crop_params(
            h_raw, w_raw, self.target_size
        )

        # Adjust intrinsics (supports batched (N, 3, 3))
        K_adj = adjust_intrinsics_manual(K_stack, scale, sy, sx)  # (N, 3, 3)

        processed_images = []
        processed_depths = []
        processed_poses = []
        processed_intrinsics = []
        original_images = []

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

            # Process depth (values in meters, only pixel grid changes)
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
            - depths: Concatenated left+right depths [2*N, 1, H, W]
            - poses: Concatenated left+right poses [2*N, 4, 4]
            - intrinsics: Concatenated left+right intrinsics [2*N, 3, 3]
            - original_images: Concatenated original images [2*N, 3, H, W]
        """
        sample_info = self.samples[index]

        left_data = self._process_side(sample_info, side="left")
        right_data = self._process_side(sample_info, side="right")

        # Normalize poses relative to first frame of left camera
        # Poses are w2c. To normalize: new_w2c = old_w2c @ inv(first_w2c) = old_w2c @ first_c2w
        first_c2w = torch.linalg.inv(left_data["poses"][0])
        left_data["poses"] = left_data["poses"] @ first_c2w
        right_data["poses"] = right_data["poses"] @ first_c2w

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
