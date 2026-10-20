#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Convert raw TartanAir (V1) downloads into the processed layout used by
``stereosplat.datasets.tartan`` (training data).

Input (``--input-dir``): a local copy of TartanAir V1 with the same layout as
the Hugging Face dataset ``theairlabcmu/tartanair`` (or the official download
scripts)::

    {input_dir}/{env}/Easy/image_left.zip
    {input_dir}/{env}/Easy/image_right.zip
    {input_dir}/{env}/Easy/depth_left.zip
    {input_dir}/{env}/Easy/depth_right.zip
    {input_dir}/{env}/Hard/...

Each ``image_*.zip`` contains ``{env}/{Easy|Hard}/P***/image_{side}/*.png`` and
``{env}/{Easy|Hard}/P***/pose_{side}.txt``; each ``depth_*.zip`` contains
``{env}/{Easy|Hard}/P***/depth_{side}/*_depth.npy``. Flow and segmentation
archives are not needed.

Output (``--output-dir``), one folder per trajectory::

    {env}_{Easy|Hard}_{P***}/
        rgb_left/000000.png ...
        rgb_right/000000.png ...
        depth_left/000000.npy ...      (480, 640) float32, meters
        depth_right/000000.npy ...
        poses_left.npy                 (N, 4, 4) float32 camera-to-world, OpenCV
        poses_right.npy
        intrinsics_left.npy            (3, 3) float32
        intrinsics_right.npy

Example::

    python scripts/prepare_data/tartan.py \\
        --input-dir /data/raw/tartanair \\
        --output-dir /data/processed/tartanair_processed \\
        --split-file data/splits/tartan/train.txt
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm

DIFFICULTY_LEVELS = ["Easy", "Hard"]
SIDES = ["left", "right"]

# Pinhole intrinsics of the 640x480 TartanAir cameras.
TARTAN_INTRINSICS = np.array(
    [
        [320.0, 0.0, 320.0],
        [0.0, 320.0, 240.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


def save_rgb_frames_as_png(
    zip_ref: zipfile.ZipFile, file_list: list[str], output_dir: Path
) -> None:
    """Decode sorted images and save as RGB 000000.png, 000001.png, ..."""
    if not file_list:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, fname in enumerate(sorted(file_list)):
        with Image.open(io.BytesIO(zip_ref.read(fname))) as img:
            img.convert("RGB").save(output_dir / f"{idx:06d}.png", "PNG")


def save_depths_as_npy(
    zip_ref: zipfile.ZipFile, file_list: list[str], output_dir: Path
) -> None:
    """Save sorted depth maps (meters) as float32 000000.npy, 000001.npy, ..."""
    if not file_list:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, fname in enumerate(sorted(file_list)):
        with io.BytesIO(zip_ref.read(fname)) as bio:
            if fname.endswith(".npy"):
                arr = np.load(bio)
            else:
                with Image.open(bio) as img:
                    arr = np.array(img)
        np.save(output_dir / f"{idx:06d}.npy", arr.astype(np.float32))


def process_pose_from_bytes(pose_bytes: bytes) -> np.ndarray | None:
    """Parse a TartanAir pose file into (N, 4, 4) camera-to-world matrices.

    Lines are ``tx ty tz qx qy qz qw`` (NED world, camera x forward). The
    cyclic permutation converts world and camera frames to OpenCV convention.
    """
    lines = [ln.strip() for ln in pose_bytes.decode("utf-8").strip().split("\n")]
    lines = [ln for ln in lines if ln]
    if not lines:
        return None
    poses = np.array([list(map(float, ln.split())) for ln in lines])

    perm = [1, 2, 0, 4, 5, 3, 6]
    poses = poses[:, perm]
    trans = poses[:, :3]
    quats = poses[:, 3:]
    cam_rot = Rotation.from_quat(quats, scalar_first=False).as_matrix()

    c2ws = np.eye(4)[None].repeat(poses.shape[0], axis=0).astype(np.float32)
    c2ws[:, :3, :3] = cam_rot
    c2ws[:, :3, 3] = trans
    return c2ws


def _in_trajectory(name: str, trajectory_id: str) -> bool:
    return (
        f"/{trajectory_id}/" in name or name.startswith(f"{trajectory_id}/")
    ) and not name.startswith("__MACOSX")


def _filter_zip_files(
    all_files: list[str], trajectory_id: str, extensions: tuple[str, ...]
) -> list[str]:
    return [
        f
        for f in all_files
        if _in_trajectory(f, trajectory_id) and f.lower().endswith(extensions)
    ]


def _process_rgb_and_poses_for_side(
    zf: zipfile.ZipFile, trajectory_id: str, side: str, out_dir: Path
) -> None:
    all_files = zf.namelist()
    rgb_dir = out_dir / f"rgb_{side}"
    poses_file = out_dir / f"poses_{side}.npy"
    intrinsics_file = out_dir / f"intrinsics_{side}.npy"

    if not rgb_dir.exists():
        img_files = _filter_zip_files(all_files, trajectory_id, (".png", ".jpg"))
        save_rgb_frames_as_png(zf, img_files, rgb_dir)

    if not poses_file.exists():
        target = f"pose_{side}.txt"
        candidates = [
            f
            for f in all_files
            if _in_trajectory(f, trajectory_id) and f.endswith(target)
        ]
        if candidates:
            poses = process_pose_from_bytes(zf.read(candidates[0]))
            if poses is not None:
                np.save(poses_file, poses)

    if not intrinsics_file.exists():
        np.save(intrinsics_file, TARTAN_INTRINSICS)


def process_trajectory(
    env_path: Path, difficulty: str, trajectory_id: str, output_root: Path
) -> str:
    """Write one processed trajectory. Existing outputs are kept (resumable)."""
    diff_path = env_path / difficulty
    traj_name = f"{env_path.name}_{difficulty}_{trajectory_id}"
    traj_dir = output_root / traj_name
    traj_dir.mkdir(parents=True, exist_ok=True)

    for side in SIDES:
        image_zip = diff_path / f"image_{side}.zip"
        if image_zip.exists():
            with zipfile.ZipFile(image_zip, "r") as zf:
                _process_rgb_and_poses_for_side(zf, trajectory_id, side, traj_dir)

    for side in SIDES:
        depth_zip = diff_path / f"depth_{side}.zip"
        depth_dir = traj_dir / f"depth_{side}"
        if depth_zip.exists() and not depth_dir.exists():
            with zipfile.ZipFile(depth_zip, "r") as zf:
                files = _filter_zip_files(
                    zf.namelist(), trajectory_id, (".npy", ".png")
                )
                save_depths_as_npy(zf, files, depth_dir)

    return traj_name


def validate_trajectory(traj_dir: Path) -> tuple[bool, str]:
    """Check num_rgb == num_depth == num_poses > 0 and intrinsics exist."""
    for side in SIDES:
        rgb_dir = traj_dir / f"rgb_{side}"
        depth_dir = traj_dir / f"depth_{side}"
        poses_file = traj_dir / f"poses_{side}.npy"
        n_rgb = len(list(rgb_dir.glob("*.png"))) if rgb_dir.exists() else 0
        n_depth = len(list(depth_dir.glob("*.npy"))) if depth_dir.exists() else 0
        n_poses = np.load(poses_file).shape[0] if poses_file.exists() else 0
        has_k = (traj_dir / f"intrinsics_{side}.npy").exists()
        if not (n_rgb == n_depth == n_poses and has_k and n_rgb > 0):
            return False, (
                f"{side}: rgb={n_rgb}, depth={n_depth}, poses={n_poses}, "
                f"intrinsics={has_k}"
            )
    return True, "OK"


def discover_trajectories(env_path: Path) -> list[tuple[str, str]]:
    """Return (difficulty, trajectory_id) pairs found in image_left.zip."""
    tasks = []
    for difficulty in DIFFICULTY_LEVELS:
        image_zip = env_path / difficulty / "image_left.zip"
        if not image_zip.exists():
            continue
        try:
            with zipfile.ZipFile(image_zip, "r") as zf:
                traj_ids = set()
                for n in zf.namelist():
                    for part in n.split("/"):
                        if re.match(r"^P\d{3}$", part):
                            traj_ids.add(part)
                            break
        except zipfile.BadZipFile:
            print(f"  Warning: corrupt zip {image_zip}")
            continue
        tasks.extend((difficulty, t) for t in sorted(traj_ids))
    return tasks


def read_split_file(path: str | None) -> set[str] | None:
    if path is None:
        return None
    with open(path) as f:
        return {ln.strip() for ln in f if ln.strip()}


def discover_envs(input_dir: Path) -> list[str]:
    return sorted(
        p.name
        for p in input_dir.iterdir()
        if p.is_dir() and any((p / d).is_dir() for d in DIFFICULTY_LEVELS)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert raw TartanAir V1 zips to the processed layout."
    )
    parser.add_argument("--input-dir", required=True, help="Raw TartanAir root.")
    parser.add_argument("--output-dir", required=True, help="Processed output root.")
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=None,
        help="Environments to process (default: every environment in --input-dir).",
    )
    parser.add_argument(
        "--split-file",
        default=None,
        help="Only process trajectories listed in this file "
        "(e.g. data/splits/tartan/train.txt).",
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=4, help="Parallel worker processes."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    wanted = read_split_file(args.split_file)

    envs = args.scenes or discover_envs(input_dir)
    tasks = []
    for env in envs:
        env_path = input_dir / env
        if not env_path.is_dir():
            print(f"  Warning: environment not found: {env_path}")
            continue
        for difficulty, traj in discover_trajectories(env_path):
            if wanted is None or f"{env}_{difficulty}_{traj}" in wanted:
                tasks.append((env_path, difficulty, traj))
    print(f"TartanAir V1: {len(tasks)} trajectories from {len(envs)} environments")

    names = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(process_trajectory, env_path, diff, traj, output_dir)
            for env_path, diff, traj in tasks
        ]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            names.append(f.result())

    invalid = []
    for name in sorted(names):
        ok, detail = validate_trajectory(output_dir / name)
        if not ok:
            invalid.append((name, detail))
    print(f"Processed {len(names)} trajectories into {output_dir}")
    if wanted is not None:
        missing = sorted(w for w in wanted if not (output_dir / w).is_dir())
        if missing:
            print(f"WARNING: {len(missing)} split entries not produced: {missing}")
    if invalid:
        print(f"WARNING: {len(invalid)} inconsistent trajectories:")
        for name, detail in invalid:
            print(f"  {name}: {detail}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
