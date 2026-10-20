# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Data utilities for dataset processing.

Helper functions for image/depth resizing, intrinsic adjustments, window
creation, and loading the processed per-frame dataset layout from local disk.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


# =============================================================================
# Image/Depth Processing Utilities
# =============================================================================


def create_overlapping_windows(
    arr: np.ndarray, window_size: int, overlap: int
) -> np.ndarray:
    arr = np.asarray(arr)
    n = arr.shape[0]

    # Special case: window_size=1 returns each frame as its own window
    if window_size == 1:
        return arr.reshape(-1, 1)

    step = window_size - overlap
    windows = []

    start = 0
    while start < n - window_size:
        windows.append(arr[start : start + window_size])
        start += step

    if not windows:
        return np.empty((0, window_size), dtype=arr.dtype)
    return np.vstack(windows)


def get_resize_crop_params(
    h: int, w: int, target_size: tuple[int, int] = (320, 640)
) -> tuple[float, int, int, int, int]:
    th, tw = target_size

    # 1. Calculate Scale: Ensure image covers the target size completely
    scale_h = th / h
    scale_w = tw / w
    scale = max(scale_h, scale_w)  # Use max to avoid black borders

    new_w = int(w * scale)
    new_h = int(h * scale)

    # 2. Calculate Center Crop (or Random Crop) coordinates
    start_x = (new_w - tw) // 2
    start_y = (new_h - th) // 2

    return scale, new_h, new_w, start_y, start_x


def process_image(
    img: np.ndarray,
    scale: float,
    new_h: int,
    new_w: int,
    start_y: int,
    start_x: int,
    target_size: tuple[int, int],
) -> np.ndarray:
    th, tw = target_size
    # Use LANCZOS4 for downscaling (high quality), LINEAR for slight upscaling
    interp = cv2.INTER_LANCZOS4 if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    return resized[start_y : start_y + th, start_x : start_x + tw]


def process_depth_or_mask(
    img: np.ndarray,
    scale: float,
    new_h: int,
    new_w: int,
    start_y: int,
    start_x: int,
    target_size: tuple[int, int],
) -> np.ndarray:
    th, tw = target_size
    # NEAREST is mandatory for masks. For depth, it prevents "flying pixels" at edges.
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return resized[start_y : start_y + th, start_x : start_x + tw]


def adjust_intrinsics_manual(
    K: np.ndarray, scale: float, start_y: int, start_x: int
) -> np.ndarray:
    K_new = K.copy()
    # 1. Scale Focal Lengths and Centers
    K_new[..., 0, 0] *= scale  # fx
    K_new[..., 1, 1] *= scale  # fy
    K_new[..., 0, 2] *= scale  # cx
    K_new[..., 1, 2] *= scale  # cy

    # 2. Adjust Optical Center for Cropping
    K_new[..., 0, 2] -= start_x
    K_new[..., 1, 2] -= start_y
    return K_new


# =============================================================================
# Individual File Loading (PNG + NPY format)
# =============================================================================


def load_rgb_frames_individual(
    rgb_dir: str,
    indices: list,
) -> np.ndarray:
    """
    Load RGB frames from individual PNG files.

    Args:
        rgb_dir: Directory containing ``%06d.png`` files.
        indices: List of frame indices to load.

    Returns:
        Numpy array of shape (N, H, W, 3) containing RGB frames as uint8.
    """
    frames = []
    for idx in indices:
        with Image.open(f"{rgb_dir}/{idx:06d}.png") as img:
            frames.append(np.array(img.convert("RGB"), dtype=np.uint8))
    return np.stack(frames, axis=0)


def load_depth_frames_individual(
    depth_dir: str,
    indices: list,
) -> np.ndarray:
    """
    Load depth frames from individual NPY files.

    Args:
        depth_dir: Directory containing ``%06d.npy`` files.
        indices: List of frame indices to load.

    Returns:
        Numpy array of shape (N, H, W) containing depth values as float32.
    """
    depths = [
        np.load(f"{depth_dir}/{idx:06d}.npy").astype(np.float32) for idx in indices
    ]
    return np.stack(depths, axis=0)


def load_poses_individual(pose_path: str, indices: list) -> np.ndarray:
    """
    Load poses from a single NPY file (individual files format).

    Args:
        pose_path: Path to the ``.npy`` pose file.
        indices: List of frame indices to extract.

    Returns:
        Numpy array of shape (N, 4, 4) containing camera poses.
    """
    poses = np.load(pose_path).astype(np.float32)
    return poses[indices]


def load_intrinsics_individual(
    intrinsics_path: str, indices: list | None = None
) -> np.ndarray:
    """
    Load intrinsics matrix from a single NPY file (individual files format).

    Args:
        intrinsics_path: Path to the ``.npy`` intrinsics file.
        indices: Optional list of frame indices. When provided and intrinsics are
                 per-frame (N_frames, 3, 3), returns intrinsics for the selected
                 frames (len(indices), 3, 3). When intrinsics are shared (3, 3),
                 broadcasts to (len(indices), 3, 3).

    Returns:
        Numpy array of shape (3, 3) if indices is None, or
        (len(indices), 3, 3) if indices is provided.
    """
    intrinsics = np.load(intrinsics_path).astype(np.float32)

    if indices is None:
        if intrinsics.ndim == 3:
            return intrinsics[0]
        return intrinsics

    if intrinsics.ndim == 3:
        return intrinsics[indices]

    # Shared intrinsics (3, 3) -> broadcast to (N, 3, 3)
    return np.broadcast_to(intrinsics, (len(indices), 3, 3)).copy()


def get_num_frames_from_poses_individual(pose_path: str) -> int:
    """
    Get the total number of frames from an individual poses file.

    Args:
        pose_path: Path to the ``.npy`` pose file.

    Returns:
        Number of frames in the sequence.
    """
    return int(np.load(pose_path, mmap_mode="r").shape[0])
