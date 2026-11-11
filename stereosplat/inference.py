# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Inference helpers for rectified stereo input (used by ``scripts/demo.py``).

Everything up to (and excluding) rasterization runs on CPU or GPU; rendering
needs CUDA because gsplat has no CPU rasterizer.

Conventions (same as the datasets): OpenCV camera axes (x right, y down,
z forward), poses are world-to-camera in the model inputs, the world frame is
the first left camera, the right camera of each pair sits at ``+baseline``
along the left camera's x axis, and views are ordered ``[L0, ..., R0, ...]``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

from stereosplat.datasets.io_utils import (
    adjust_intrinsics_manual,
    get_resize_crop_params,
    process_image,
)
from stereosplat.models.build import build_model, load_checkpoint, read_checkpoint
from stereosplat.models.stereosplat import StereoSplat
from stereosplat.rendering.gaussians import Gaussians
from stereosplat.utils.preprocessing import get_transforms

IMAGE_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp")
DEFAULT_IMG_SIZE: tuple[int, int] = (320, 640)


@dataclass
class StereoInputs:
    """Model inputs for one stereo window (batch size 1).

    Attributes:
        images: [1, 2P, 3, H, W] normalized images.
        original_images: [1, 2P, 3, H, W] RGB in [0, 1] after resize/crop.
        poses: [1, 2P, 4, 4] world-to-camera.
        intrinsics: [1, 2P, 3, 3] pixel intrinsics after resize/crop.
        frame_names: names of the P left frames.
    """

    images: torch.Tensor
    original_images: torch.Tensor
    poses: torch.Tensor
    intrinsics: torch.Tensor
    frame_names: list[str]

    @property
    def num_pairs(self) -> int:
        return self.images.shape[1] // 2

    @property
    def image_size(self) -> tuple[int, int]:
        h, w = self.images.shape[-2:]
        return int(h), int(w)

    def as_model_input(self, device: torch.device | str) -> dict[str, torch.Tensor]:
        return {
            "images": self.images.to(device),
            "poses": self.poses.to(device),
            "intrinsics": self.intrinsics.to(device),
        }


def require_cuda(what: str = "Rendering") -> None:
    """Raise a clear error if CUDA is unavailable (gsplat has no CPU path)."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"{what} requires a CUDA GPU: gsplat only provides CUDA rasterization "
            "kernels. Run on a machine with CUDA and a CUDA build of PyTorch, or "
            "skip rendering (scripts/demo.py --no-render) to only export the .ply."
        )


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------


def list_images(directory: str | os.PathLike[str]) -> list[Path]:
    paths = sorted(
        p
        for p in Path(directory).iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise FileNotFoundError(
            f"No images ({', '.join(IMAGE_EXTENSIONS)}) in {directory}"
        )
    return paths


def find_stereo_sequence(directory: str | os.PathLike[str]) -> list[tuple[Path, Path]]:
    """Pair ``<directory>/left/*`` with ``<directory>/right/*`` by sorted order."""
    root = Path(directory)
    left, right = list_images(root / "left"), list_images(root / "right")
    if len(left) != len(right):
        raise ValueError(
            f"{root}: {len(left)} left but {len(right)} right images; the "
            "left/ and right/ folders must contain the same number of frames."
        )
    return list(zip(left, right))


def load_rgb(path: str | os.PathLike[str]) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_poses(path: str | os.PathLike[str]) -> np.ndarray:
    """Load left-camera camera-to-world poses as ``(N, 4, 4)`` float64.

    Accepts ``.npy`` with shape ``(N, 4, 4)``, ``(N, 3, 4)``, ``(N, 12)`` or
    ``(N, 16)``, or a text file with 12 (KITTI style, row-major 3x4) or 16
    values per line.
    """
    path = Path(path)
    arr = np.load(path) if path.suffix == ".npy" else np.loadtxt(path, ndmin=2)
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[1] in (12, 16):
        arr = arr.reshape(arr.shape[0], -1, 4)
    if arr.ndim != 3 or arr.shape[1:] not in ((3, 4), (4, 4)):
        raise ValueError(
            f"{path}: expected (N, 4, 4) or (N, 3, 4) poses, got {arr.shape}"
        )
    if arr.shape[1] == 3:
        bottom = np.broadcast_to(np.array([0.0, 0.0, 0.0, 1.0]), (arr.shape[0], 1, 4))
        arr = np.concatenate([arr, bottom], axis=1)
    return arr


def stereo_world_to_camera(left_c2w: np.ndarray, baseline: float) -> np.ndarray:
    """World-to-camera poses ``[2P, 4, 4]`` for P rectified stereo pairs.

    The left camera-to-world poses are first expressed relative to the first
    left camera; each right camera is its left camera shifted by ``baseline``
    meters along the left x axis.
    """
    if baseline <= 0:
        raise ValueError(f"baseline must be positive (meters), got {baseline}")
    left_c2w = np.asarray(left_c2w, dtype=np.float64)
    left_c2w = np.linalg.inv(left_c2w[0]) @ left_c2w
    left_w2c = np.linalg.inv(left_c2w)
    shift = np.eye(4)
    shift[0, 3] = -baseline  # right_w2c = T(-b e_x) @ left_w2c
    right_w2c = shift @ left_w2c
    return np.concatenate([left_w2c, right_w2c], axis=0)


def prepare_stereo_inputs(
    pairs: Sequence[tuple[np.ndarray, np.ndarray]],
    intrinsics: Sequence[float],
    baseline: float,
    left_c2w: np.ndarray | None = None,
    img_size: tuple[int, int] = DEFAULT_IMG_SIZE,
    frame_names: Sequence[str] | None = None,
) -> StereoInputs:
    """Resize/crop images like the datasets and build the model inputs.

    Args:
        pairs: P (left, right) RGB uint8 images of identical size.
        intrinsics: (fx, fy, cx, cy) in pixels of the input images (left
            camera; the rectified right camera shares them).
        baseline: Stereo baseline in meters.
        left_c2w: Optional (P, 4, 4) left camera-to-world poses (required if
            P > 1). Only relative poses matter; translations must be metric.
        img_size: Network input size (H, W), multiple of 32.
        frame_names: Optional names of the frames.
    """
    num_pairs = len(pairs)
    if num_pairs == 0:
        raise ValueError("At least one stereo pair is required.")
    if left_c2w is None:
        if num_pairs > 1:
            raise ValueError("Camera poses are required for more than one stereo pair.")
        left_c2w = np.eye(4)[None]
    if len(left_c2w) != num_pairs:
        raise ValueError(f"Got {len(left_c2w)} poses for {num_pairs} stereo pairs.")
    if img_size[0] % 32 or img_size[1] % 32:
        raise ValueError(f"img_size must be a multiple of 32, got {img_size}")

    h_raw, w_raw = pairs[0][0].shape[:2]
    for left, right in pairs:
        if left.shape != (h_raw, w_raw, 3) or right.shape != (h_raw, w_raw, 3):
            raise ValueError("All left/right images must be RGB with the same size.")

    fx, fy, cx, cy = (float(v) for v in intrinsics)
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    scale, new_h, new_w, sy, sx = get_resize_crop_params(h_raw, w_raw, img_size)
    K_adj = adjust_intrinsics_manual(K, scale, sy, sx)

    transform = get_transforms()
    images, originals = [], []
    for side in (0, 1):  # all left views, then all right views
        for pair in pairs:
            img = process_image(pair[side], scale, new_h, new_w, sy, sx, img_size)
            if img.shape[:2] != tuple(img_size):
                raise ValueError(
                    f"Resized image has shape {img.shape[:2]}, expected {img_size}."
                )
            images.append(transform(img))
            originals.append(
                torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
            )

    w2c = stereo_world_to_camera(left_c2w, baseline)
    n = 2 * num_pairs
    return StereoInputs(
        images=torch.stack(images)[None],
        original_images=torch.stack(originals)[None],
        poses=torch.from_numpy(w2c).float()[None],
        intrinsics=torch.from_numpy(K_adj).float().expand(1, n, 3, 3).contiguous(),
        frame_names=list(frame_names)
        if frame_names
        else [str(i) for i in range(num_pairs)],
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def load_model_config(
    checkpoint_path: str | os.PathLike[str], config_path: str | None = None
) -> dict[str, Any]:
    """Model config of a checkpoint.

    Release checkpoints carry their model config, and so do training
    checkpoints written by ``scripts/train.py`` (``model_config``). Otherwise,
    or if ``config_path`` is given, the ``model`` section (and ``max_depth``)
    of ``config_path`` is used.
    """
    ckpt = read_checkpoint(checkpoint_path)
    if isinstance(ckpt, Mapping) and "config" in ckpt and config_path is None:
        return dict(ckpt["config"])
    if isinstance(ckpt, Mapping) and "model_config" in ckpt and config_path is None:
        model_cfg = dict(ckpt["model_config"])
        model_cfg.pop("fs_checkpoint_path", None)  # path on the training machine
        return model_cfg
    if config_path is None:
        raise ValueError(
            f"{checkpoint_path} has no embedded model config; pass the training "
            "config YAML (--config)."
        )
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=False)
    assert isinstance(cfg, dict)
    model_cfg = dict(cfg["model"])
    model_cfg.setdefault("max_depth", cfg.get("max_depth", 80.0))
    model_cfg.pop("fs_checkpoint_path", None)  # may reference unset env variables
    return model_cfg


def build_inference_model(
    checkpoint_path: str | os.PathLike[str],
    fs_checkpoint_path: str | os.PathLike[str],
    config_path: str | None = None,
    device: torch.device | str = "cpu",
    backbone: torch.nn.Module | None = None,
) -> StereoSplat:
    """Build StereoSplat and load a release or training checkpoint (eval mode)."""
    model_cfg = load_model_config(checkpoint_path, config_path)
    model_cfg["fs_checkpoint_path"] = str(fs_checkpoint_path)
    model_cfg.setdefault("max_depth", 80.0)
    model = build_model(model_cfg, backbone=backbone)
    load_checkpoint(model, str(checkpoint_path))
    return model.to(device).eval()


@torch.no_grad()
def predict_gaussians(
    model: StereoSplat, inputs: StereoInputs, device: torch.device | str = "cpu"
) -> tuple[Gaussians, torch.Tensor]:
    """Run the model and return merged world-space Gaussians and per-view depths.

    Returns:
        gaussians: Gaussians of the single scene (batch size 1, on ``device``).
        depths: [2P, H, W] predicted depth (meters) of the input views.
    """
    data = inputs.as_model_input(device)
    out = model(data)
    gaussians = model.gaussian_adapter(
        extrinsics=data["poses"],
        intrinsics=data["intrinsics"],
        depths=out["depths"],
        raw_gaussians=out["raw_gaussians"],
    )
    gaussians = model.aggregate_gaussians(gaussians)
    return gaussians, out["depths"][0, :, 0]


# ---------------------------------------------------------------------------
# Novel-view cameras
# ---------------------------------------------------------------------------


def look_at_c2w(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    """OpenCV camera-to-world pose at ``position`` looking at ``target`` (y down)."""
    forward = target - position
    forward = forward / np.linalg.norm(forward)
    down = np.array([0.0, 1.0, 0.0])
    right = np.cross(down, forward)
    if np.linalg.norm(right) < 1e-8:  # looking straight up/down
        right = np.array([1.0, 0.0, 0.0])
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right, down, forward, position
    return c2w


def orbit_cameras(
    num_views: int,
    radius: float,
    target_depth: float,
    forward: float = 0.0,
) -> np.ndarray:
    """World-to-camera poses ``[num_views + 1, 4, 4]`` for a small orbit.

    View 0 is the first input (left) camera. The other views move on a circle
    of ``radius`` meters in the image plane of that camera, shifted ``forward``
    meters along its optical axis, all looking at the point ``target_depth``
    meters in front of it.
    """
    if num_views < 0 or radius < 0 or target_depth <= forward:
        raise ValueError("Need num_views >= 0, radius >= 0 and target_depth > forward.")
    target = np.array([0.0, 0.0, target_depth])
    poses = [np.eye(4)]
    for k in range(num_views):
        angle = 2.0 * np.pi * k / max(num_views, 1)
        position = np.array([radius * np.cos(angle), radius * np.sin(angle), forward])
        poses.append(np.linalg.inv(look_at_c2w(position, target)))
    return np.stack(poses)


def scene_target_depth(depths: torch.Tensor, max_depth: float = 80.0) -> float:
    """Median valid depth of the first input view (orbit look-at distance)."""
    d = depths[0]
    d = d[(d > 0) & (d < max_depth)]
    return float(d.median()) if d.numel() else 0.5 * max_depth


# ---------------------------------------------------------------------------
# Rendering (CUDA only)
# ---------------------------------------------------------------------------


@torch.no_grad()
def render_views(
    gaussians: Gaussians,
    w2c: np.ndarray | torch.Tensor,
    intrinsics: torch.Tensor,
    image_size: tuple[int, int],
) -> dict[str, torch.Tensor]:
    """Render ``[V, 4, 4]`` world-to-camera views; returns colors [V, 3, H, W],
    depths [V, H, W] and alphas [V, H, W] on CPU."""
    require_cuda()
    from stereosplat.rendering.renderer import run_renderer_in_chunk_w_trj_mode

    device = torch.device("cuda")
    gaussians = Gaussians(
        means=gaussians.means.to(device),
        scales=gaussians.scales.to(device),
        rotations=gaussians.rotations.to(device),
        harmonics=gaussians.harmonics.to(device),
        opacities=gaussians.opacities.to(device),
    )
    w2c_t = torch.as_tensor(np.asarray(w2c), dtype=torch.float32, device=device)[None]
    K = intrinsics.reshape(1, 1, 3, 3).to(device).expand(1, w2c_t.shape[1], 3, 3)
    out = run_renderer_in_chunk_w_trj_mode(
        gaussians=gaussians,
        extrinsics=w2c_t,
        intrinsics=K.contiguous(),
        image_shape=image_size,
    )
    return {
        "colors": out["colors"][0].clamp(0.0, 1.0).cpu(),
        "depths": out["depths"][0].cpu(),
        "alphas": out["alphas"][0].cpu(),
    }
