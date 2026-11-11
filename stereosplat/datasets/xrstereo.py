# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
XRStereo dataset (training).

Reads the extracted XRStereo release (https://huggingface.co/datasets/facebook/xrstereo)
from local disk. No conversion is needed: the loader reads the released files
directly.

Data format (after extracting the tar shards):
    {dataset_path}/{sequence}/
        calibration.json   {"f", "cx", "cy", "baseline_m"}, shared by both cameras
        poses_left.npy     (N, 4, 4) float64 world-from-camera of the left camera,
                           metres, Blender camera axes (x right, y up, z backward)
        frame_0000.npz     image_left, image_right: (H, W, 3) uint8
        ...                disparity_left, disparity_right: (H, W) float16, pixels

The cameras are rectified pinhole cameras with the same intrinsics; the right
camera is the left camera translated by ``baseline_m`` along its x axis.
Metric depth is ``f * baseline_m / disparity``; disparity <= 0 is invalid and
becomes depth 0 (excluded from the depth loss by the ``(0, max_depth)`` mask).
"""

from __future__ import annotations

import json
from collections.abc import Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from stereosplat.datasets.io_utils import (
    adjust_intrinsics_manual,
    create_overlapping_windows,
    get_num_frames_from_poses_individual,
    get_resize_crop_params,
    process_depth_or_mask,
    process_image,
    report_missing_scenes,
)

# Blender camera axes (x right, y up, z backward) -> OpenCV (x right, y down,
# z forward), applied on the right of a camera-to-world matrix.
BLENDER_TO_OPENCV: np.ndarray = np.diag([1.0, -1.0, -1.0, 1.0])

SIDES: tuple[str, str] = ("left", "right")


def load_calibration(path: str) -> tuple[np.ndarray, float]:
    """Return the shared pinhole intrinsics ``K`` (3, 3) and the baseline (m)."""
    with open(path) as f:
        calib = json.load(f)
    fx = float(calib["f"])
    K = np.array(
        [[fx, 0.0, float(calib["cx"])], [0.0, fx, float(calib["cy"])], [0, 0, 1]],
        dtype=np.float32,
    )
    return K, float(calib["baseline_m"])


def disparity_to_depth(disparity: np.ndarray, f: float, baseline: float) -> np.ndarray:
    """Metric depth (float32, metres); 0 where the disparity is <= 0."""
    d = disparity.astype(np.float32)
    fb = np.float32(f * baseline)
    return np.where(d > 0, fb / np.maximum(d, 1e-8), 0.0).astype(np.float32)


def stereo_camera_to_world(
    pose_left_blender: np.ndarray, baseline: float
) -> tuple[np.ndarray, np.ndarray]:
    """Left and right camera-to-world matrices in OpenCV axes (float64)."""
    c2w_left = pose_left_blender.astype(np.float64) @ BLENDER_TO_OPENCV
    left_to_right = np.eye(4)
    left_to_right[0, 3] = baseline
    return c2w_left, c2w_left @ left_to_right


class XRStereoDataset(Dataset):
    """
    XRStereo dataset that reads the extracted release from local disk.

    Each sequence has 5 stereo frames. Windows are built like for the other
    datasets (``num_views`` frames with an overlap of 1), which gives one
    window per sequence (frames 0-2) for ``num_views=3, stride=1``, the
    setting used to train the released model.

    Args:
        path: Directory that contains the sequence folders (``sequences/``
            of the extracted release).
        target_size: Target image size as (height, width) tuple
        transforms: Torchvision transforms to apply to images
        scene_names: List of sequence names to include
        num_views: Number of views per window
        stride: Stride for frame sampling
        num_samples: Maximum number of samples (0 for all)
        allow_missing_scenes: Skip split sequences whose poses cannot be
            loaded (with a warning) instead of raising ``FileNotFoundError``.
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

        self.scene_names = scene_names
        self.samples = []
        self.missing_scenes: list[str] = []
        self._create_samples()
        report_missing_scenes(
            type(self).__name__, self.root, self.missing_scenes, allow_missing_scenes
        )

    def _create_samples(self) -> None:
        """Create sample windows from all sequences."""
        for scene_name in self.scene_names:
            scene_path = f"{self.root}/{scene_name}"
            poses_path = f"{scene_path}/poses_left.npy"

            try:
                num_frames = get_num_frames_from_poses_individual(poses_path)
            except Exception as e:
                self.missing_scenes.append(f"{scene_name}: {poses_path} ({e})")
                continue

            strided_indices = np.arange(num_frames)[:: self.stride]
            windows = create_overlapping_windows(
                strided_indices, window_size=self.num_views, overlap=1
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

    def __getitem__(self, index: int) -> dict:
        """
        Get a sample from the dataset.

        Returns:
            Dictionary containing:
            - seq_name: Sequence name
            - frame_number: List of frame indices
            - images: Concatenated left+right images [2*N, 3, H, W]
            - depths: Concatenated left+right depths [2*N, 1, H, W]
            - poses: Concatenated left+right w2c poses [2*N, 4, 4], relative
              to the first left camera
            - intrinsics: Concatenated left+right intrinsics [2*N, 3, 3]
            - original_images: Concatenated images in [0, 1] [2*N, 3, H, W]
        """
        sample_info = self.samples[index]
        scene_path = sample_info["scene_path"]
        indices = sample_info["frame_indices"].tolist()

        K, baseline = load_calibration(f"{scene_path}/calibration.json")
        poses_left = np.load(f"{scene_path}/poses_left.npy", allow_pickle=False)

        out: dict[str, dict[str, list[torch.Tensor]]] = {
            side: {
                "images": [],
                "original_images": [],
                "depths": [],
                "poses": [],
                "intrinsics": [],
            }
            for side in SIDES
        }
        resize = None
        K_adj = None
        for idx in indices:
            # Each frame file holds both views; read it once.
            with np.load(f"{scene_path}/frame_{idx:04d}.npz", allow_pickle=False) as z:
                frame = {key: z[key] for key in z.files}
            if resize is None:
                h_raw, w_raw = frame["image_left"].shape[:2]
                resize = get_resize_crop_params(h_raw, w_raw, self.target_size)
                scale, _, _, sy, sx = resize
                K_adj = torch.from_numpy(adjust_intrinsics_manual(K, scale, sy, sx))
            scale, new_h, new_w, sy, sx = resize

            c2w = dict(zip(SIDES, stereo_camera_to_world(poses_left[idx], baseline)))
            for side in SIDES:
                img = process_image(
                    frame[f"image_{side}"],
                    scale,
                    new_h,
                    new_w,
                    sy,
                    sx,
                    self.target_size,
                )
                out[side]["images"].append(self.transforms(img))
                out[side]["original_images"].append(
                    torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
                )
                # Depth at full resolution, then nearest resize (as the other
                # datasets, which store depth maps).
                depth = disparity_to_depth(
                    frame[f"disparity_{side}"], K[0, 0], baseline
                )
                depth = process_depth_or_mask(
                    depth, scale, new_h, new_w, sy, sx, self.target_size
                )
                out[side]["depths"].append(torch.from_numpy(depth).unsqueeze(0))
                w2c = np.linalg.inv(c2w[side]).astype(np.float32)
                out[side]["poses"].append(torch.from_numpy(w2c))
                out[side]["intrinsics"].append(K_adj.clone())

        stacked = {
            side: {key: torch.stack(values) for key, values in out[side].items()}
            for side in SIDES
        }
        # Normalize poses relative to the first left camera (w2c @ first_c2w).
        first_c2w = torch.linalg.inv(stacked["left"]["poses"][0])
        for side in SIDES:
            stacked[side]["poses"] = stacked[side]["poses"] @ first_c2w

        result = {
            "seq_name": sample_info["seq_name"],
            "frame_number": indices,
        }
        for key in ("images", "depths", "poses", "intrinsics", "original_images"):
            result[key] = torch.cat(
                [stacked["left"][key], stacked["right"][key]], dim=0
            )
        return result
