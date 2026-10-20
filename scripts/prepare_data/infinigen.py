#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Convert Infinigen stereo scenes into the processed layout used by
``stereosplat.datasets.infinigen``.

Input (``--input-dir``): a directory containing Infinigen scene folders in the
standard Infinigen frame-export layout, e.g. the InfinigenSV release
(Hugging Face dataset ``MatchLab/Infinigen``) or scenes rendered with
Infinigen yourself. Scene folders are found at any depth below
``--input-dir`` (``{input_dir}/{scene}`` or ``{input_dir}/{train|val|test}/{scene}``)::

    {scene}/frames/Image/camera_0/*.png     left RGB
    {scene}/frames/Image/camera_1/*.png     right RGB
    {scene}/frames/Depth/camera_{0,1}/*.npy depth (meters)
    {scene}/frames/camview/camera_{0,1}/*.npz  'T' (Blender camera-to-world), 'K'

Output (``--output-dir``), one folder per scene (named after the scene folder)::

    {scene}/
        rgb_left/000000.png ...
        rgb_right/000000.png ...
        depth_left/000000.npy ...       (H, W) float32, meters
        depth_right/000000.npy ...
        poses_left.npy                  (N, 4, 4) float32 camera-to-world
        poses_right.npy
        intrinsics_left.npy             (N, 3, 3) float32, per frame
        intrinsics_right.npy

Example::

    python scripts/prepare_data/infinigen.py \\
        --input-dir /data/raw/InfinigenSV \\
        --output-dir /data/processed/infinigen_processed \\
        --split-file data/splits/infinigen_static/test.txt
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

CAM_MAP = {"left": "camera_0", "right": "camera_1"}

# Pre-multiplied to Blender camera-to-world poses: converts the world frame
# from Blender (Y up) to OpenCV (Y down) so that poses and depth stay consistent.
WORLD_TRANSFORM = np.array(
    [
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
    ],
    dtype=np.float32,
)


def save_rgb_frames_as_png(image_paths: list[str], output_dir: Path) -> None:
    if not image_paths:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, img_path in enumerate(sorted(image_paths)):
        with Image.open(img_path) as img:
            img.convert("RGB").save(output_dir / f"{idx:06d}.png", "PNG")


def save_depths_as_npy(depth_paths: list[str], output_dir: Path) -> None:
    if not depth_paths:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, p in enumerate(sorted(depth_paths)):
        np.save(output_dir / f"{idx:06d}.npy", np.load(p).astype(np.float32))


def process_camera_data(
    cam_paths: list[str], output_pose_path: Path, output_intr_path: Path
) -> None:
    """Stack per-frame camview .npz files into poses (N, 4, 4) and K (N, 3, 3)."""
    if not cam_paths:
        return
    poses_list = []
    intrinsics_list = []
    for p in sorted(cam_paths):
        with np.load(p) as data:
            blender_c2w = data["T"].astype(np.float32)
            K = data["K"].astype(np.float32)
        poses_list.append(WORLD_TRANSFORM @ blender_c2w)
        intrinsics_list.append(K)
    np.save(output_pose_path, np.stack(poses_list).astype(np.float32))
    np.save(output_intr_path, np.stack(intrinsics_list).astype(np.float32))


def process_infinigen_scene(scene_path: Path, output_dir: Path) -> list[str]:
    """Write one processed scene; returns error messages (empty on success)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    errors = []
    frames = scene_path / "frames"
    for side, cam_folder in CAM_MAP.items():
        img_files = sorted(glob.glob(str(frames / "Image" / cam_folder / "*.png")))
        depth_files = sorted(glob.glob(str(frames / "Depth" / cam_folder / "*.npy")))
        cam_files = sorted(glob.glob(str(frames / "camview" / cam_folder / "*.npz")))

        rgb_dir = output_dir / f"rgb_{side}"
        if img_files and not rgb_dir.exists():
            save_rgb_frames_as_png(img_files, rgb_dir)

        depth_dir = output_dir / f"depth_{side}"
        if depth_files and not depth_dir.exists():
            save_depths_as_npy(depth_files, depth_dir)

        pose_out = output_dir / f"poses_{side}.npy"
        intr_out = output_dir / f"intrinsics_{side}.npy"
        if cam_files and (not pose_out.exists() or not intr_out.exists()):
            process_camera_data(cam_files, pose_out, intr_out)

        if not (img_files and depth_files and cam_files):
            errors.append(
                f"{side}: images={len(img_files)}, depths={len(depth_files)}, "
                f"cameras={len(cam_files)}"
            )
        elif not len(img_files) == len(depth_files) == len(cam_files):
            errors.append(
                f"{side}: count mismatch images={len(img_files)}, "
                f"depths={len(depth_files)}, cameras={len(cam_files)}"
            )
    return errors


def discover_scenes(input_dir: Path) -> dict[str, Path]:
    """Map scene name -> scene folder for every folder containing frames/Image."""
    scenes: dict[str, Path] = {}
    for image_dir in sorted(input_dir.glob("**/frames/Image")):
        scene_path = image_dir.parent.parent
        if scene_path.name in scenes:
            print(f"  Warning: duplicate scene name {scene_path.name}; keeping first")
            continue
        scenes[scene_path.name] = scene_path
    return scenes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Infinigen scenes to the processed layout."
    )
    parser.add_argument("--input-dir", required=True, help="Root of Infinigen scenes.")
    parser.add_argument("--output-dir", required=True, help="Processed output root.")
    parser.add_argument(
        "--scenes", nargs="+", default=None, help="Scene names to process."
    )
    parser.add_argument(
        "--split-file",
        default=None,
        help="Only process scenes listed in this file "
        "(e.g. data/splits/infinigen_static/test.txt).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    available = discover_scenes(input_dir)
    wanted = list(available)
    if args.scenes:
        wanted = args.scenes
    if args.split_file:
        with open(args.split_file) as f:
            split = [ln.strip() for ln in f if ln.strip()]
        wanted = [s for s in split if args.scenes is None or s in args.scenes]

    missing = [s for s in wanted if s not in available]
    todo = [s for s in wanted if s in available]
    print(f"Infinigen: {len(todo)} scenes to process ({len(available)} found)")

    failed = []
    for name in tqdm(todo, desc="Processing"):
        errors = process_infinigen_scene(available[name], output_dir / name)
        if errors:
            failed.append((name, errors))

    print(f"Processed {len(todo)} scenes into {output_dir}")
    if missing:
        print(f"WARNING: {len(missing)} requested scenes not found: {missing}")
    for name, errors in failed:
        print(f"WARNING: {name}: {'; '.join(errors)}")
    return 1 if (missing or failed) else 0


if __name__ == "__main__":
    sys.exit(main())
