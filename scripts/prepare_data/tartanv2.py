#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Convert raw TartanAir V2 downloads into the processed layout used by
``stereosplat.datasets.tartanv2``.

Input (``--input-dir``): a local copy of the TartanAir V2 release with the same
layout as the Hugging Face dataset ``theairlabcmu/tartanair2``::

    {input_dir}/{Env}/Data_easy/image_lcam_front.zip
    {input_dir}/{Env}/Data_easy/image_rcam_front.zip
    {input_dir}/{Env}/Data_easy/depth_lcam_front.zip
    {input_dir}/{Env}/Data_easy/depth_rcam_front.zip
    {input_dir}/{Env}/Data_hard/...

Each ``image_*.zip`` contains ``{Env}/Data_*/P***/image_{cam}/*.png`` and the
pose file ``{Env}/Data_*/P***/pose_{cam}.txt``; each ``depth_*.zip`` contains
``{Env}/Data_*/P***/depth_{cam}/*.png`` (float32 depth packed into RGBA PNGs).

Output (``--output-dir``), one folder per trajectory::

    {Env}_{easy|hard}_{P***}/
        rgb_lcam_front/000000.png ...       (copied byte-for-byte)
        rgb_rcam_front/000000.png ...
        depth_lcam_front/000000.npy ...     (H, W) float32, meters
        depth_rcam_front/000000.npy ...
        poses_lcam_front.npy                (N, 4, 4) float32 camera-to-world, OpenCV
        poses_rcam_front.npy
        intrinsics_lcam_front.npy           (3, 3) float32
        intrinsics_rcam_front.npy

Example::

    python scripts/prepare_data/tartanv2.py \\
        --input-dir /data/raw/tartanair_v2 \\
        --output-dir /data/processed/tartanair_v2_processed \\
        --split-file data/splits/tartanv2/test.txt
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm

ALL_CAMERA_DIRECTIONS = [
    "front",
    "back",
    "left",
    "right",
    "top",
    "bottom",
    "fish",
    "equirect",
]
DEFAULT_CAMERA_DIRECTIONS = ["front"]
CAMERA_RIGS = ["lcam", "rcam"]
DIFFICULTY_LEVELS = ["Data_easy", "Data_hard"]

# Pinhole intrinsics of the 640x640 TartanAir V2 cameras.
TARTANV2_PINHOLE_INTRINSICS = np.array(
    [
        [320.0, 0.0, 320.0],
        [0.0, 320.0, 320.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


def save_rgb_frames_as_png(
    zip_ref: zipfile.ZipFile, file_list: list[str], output_dir: Path
) -> None:
    """Write sorted images as 000000.png, 000001.png, ... (PNG bytes are copied)."""
    if not file_list:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, fname in enumerate(sorted(file_list)):
        output_path = output_dir / f"{idx:06d}.png"
        img_data = zip_ref.read(fname)
        if fname.lower().endswith(".png"):
            output_path.write_bytes(img_data)
        else:
            with Image.open(io.BytesIO(img_data)) as img:
                img.convert("RGB").save(output_path, "PNG")


def save_depths_as_npy(
    zip_ref: zipfile.ZipFile, file_list: list[str], output_dir: Path
) -> None:
    """Decode depth files to (H, W) float32 .npy files.

    TartanAir V2 stores float32 depth as 4-channel PNGs whose 4 bytes per pixel
    are the raw bytes of the float. cv2 returns them in BGRA order, which is
    reinterpreted as little-endian float32.
    """
    if not file_list:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, fname in enumerate(sorted(file_list)):
        data = zip_ref.read(fname)
        if fname.endswith(".npy"):
            with io.BytesIO(data) as bio:
                depth = np.load(bio).astype(np.float32)
        else:
            arr = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
            if img is not None and img.ndim == 3 and img.shape[2] == 4:
                depth = img.view("<f4").squeeze(axis=-1)
            else:
                with io.BytesIO(data) as bio, Image.open(bio) as pil_img:
                    depth = np.array(pil_img).astype(np.float32)
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        np.save(output_dir / f"{idx:06d}.npy", depth.astype(np.float32))


def process_pose_from_bytes(pose_bytes: bytes) -> np.ndarray | None:
    """Parse a TartanAir pose file into (N, 4, 4) camera-to-world matrices.

    Each line is ``tx ty tz qx qy qz qw`` in the NED world frame with the
    TartanAir camera frame (x forward, y right, z down). A cyclic permutation of
    translation and quaternion components converts both the world frame
    (NED -> EDN) and the camera frame (to OpenCV x right, y down, z forward):
    ``c2w_cv = T_ned2cv @ c2w_ned @ T_cam2cv``.
    """
    lines = [ln.strip() for ln in pose_bytes.decode("utf-8").strip().split("\n")]
    lines = [ln for ln in lines if ln]
    if not lines:
        return None
    poses = np.array([list(map(float, ln.split())) for ln in lines])
    if poses.shape[1] != 7:
        print(f"  Warning: expected 7 columns in pose file, got {poses.shape[1]}")
        return None

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


def _process_rgb_and_poses_for_camera(
    zf: zipfile.ZipFile,
    trajectory_id: str,
    camera_name: str,
    out_dir: Path,
) -> None:
    all_files = zf.namelist()
    rgb_dir = out_dir / f"rgb_{camera_name}"
    poses_file = out_dir / f"poses_{camera_name}.npy"
    intrinsics_file = out_dir / f"intrinsics_{camera_name}.npy"

    if not rgb_dir.exists():
        img_files = _filter_zip_files(all_files, trajectory_id, (".png", ".jpg"))
        save_rgb_frames_as_png(zf, img_files, rgb_dir)

    if not poses_file.exists():
        target = f"pose_{camera_name}.txt"
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
        np.save(intrinsics_file, TARTANV2_PINHOLE_INTRINSICS)


def process_trajectory(
    diff_path: Path,
    trajectory_id: str,
    traj_dir: Path,
    camera_directions: list[str],
) -> None:
    """Write one processed trajectory. Existing outputs are kept (resumable)."""
    traj_dir.mkdir(parents=True, exist_ok=True)
    for rig in CAMERA_RIGS:
        for direction in camera_directions:
            camera_name = f"{rig}_{direction}"
            image_zip = diff_path / f"image_{camera_name}.zip"
            if image_zip.exists():
                with zipfile.ZipFile(image_zip, "r") as zf:
                    _process_rgb_and_poses_for_camera(
                        zf, trajectory_id, camera_name, traj_dir
                    )
            else:
                print(f"  Warning: missing {image_zip}")

            depth_zip = diff_path / f"depth_{camera_name}.zip"
            depth_dir = traj_dir / f"depth_{camera_name}"
            if depth_zip.exists() and not depth_dir.exists():
                with zipfile.ZipFile(depth_zip, "r") as zf:
                    files = _filter_zip_files(
                        zf.namelist(), trajectory_id, (".npy", ".png")
                    )
                    save_depths_as_npy(zf, files, depth_dir)
            elif not depth_zip.exists():
                print(f"  Warning: missing {depth_zip}")


def validate_trajectory(
    traj_dir: Path, camera_directions: list[str]
) -> tuple[bool, str]:
    """Check num_rgb == num_depth == num_poses > 0 and intrinsics exist."""
    for rig in CAMERA_RIGS:
        for direction in camera_directions:
            cam = f"{rig}_{direction}"
            rgb_dir = traj_dir / f"rgb_{cam}"
            depth_dir = traj_dir / f"depth_{cam}"
            poses_file = traj_dir / f"poses_{cam}.npy"
            n_rgb = len(list(rgb_dir.glob("*.png"))) if rgb_dir.exists() else 0
            n_depth = len(list(depth_dir.glob("*.npy"))) if depth_dir.exists() else 0
            n_poses = np.load(poses_file).shape[0] if poses_file.exists() else 0
            has_k = (traj_dir / f"intrinsics_{cam}.npy").exists()
            if not (n_rgb == n_depth == n_poses and has_k and n_rgb > 0):
                return False, (
                    f"{cam}: rgb={n_rgb}, depth={n_depth}, poses={n_poses}, "
                    f"intrinsics={has_k}"
                )
    return True, "OK"


def discover_trajectories(
    env_path: Path, camera_directions: list[str]
) -> list[tuple[str, str]]:
    """Return (difficulty_dir, trajectory_id) pairs found in the image zips."""
    probe_camera = f"lcam_{camera_directions[0]}"
    tasks = []
    for difficulty in DIFFICULTY_LEVELS:
        image_zip = env_path / difficulty / f"image_{probe_camera}.zip"
        if not image_zip.exists():
            continue
        try:
            with zipfile.ZipFile(image_zip, "r") as zf:
                traj_ids = set()
                for n in zf.namelist():
                    for part in n.split("/"):
                        if re.match(r"^P\d{3,}$", part):
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
        description="Convert raw TartanAir V2 zips to the processed layout.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input-dir", required=True, help="Raw TartanAir V2 root.")
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
        "(e.g. data/splits/tartanv2/test.txt).",
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=DEFAULT_CAMERA_DIRECTIONS,
        choices=ALL_CAMERA_DIRECTIONS,
        help="Camera directions to process (default: front).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    wanted = read_split_file(args.split_file)
    envs = args.scenes or discover_envs(input_dir)
    if wanted is not None:
        wanted_envs = {w.rsplit("_", 2)[0] for w in wanted}
        envs = [e for e in envs if e in wanted_envs]

    print(f"TartanAir V2: {len(envs)} environments, cameras {args.cameras}")
    invalid: list[tuple[str, str]] = []
    n_done = 0
    for env in envs:
        env_path = input_dir / env
        if not env_path.is_dir():
            print(f"  Warning: environment not found: {env_path}")
            continue
        for difficulty, traj in tqdm(
            discover_trajectories(env_path, args.cameras), desc=env
        ):
            traj_name = f"{env}_{difficulty.replace('Data_', '')}_{traj}"
            if wanted is not None and traj_name not in wanted:
                continue
            traj_dir = output_dir / traj_name
            process_trajectory(env_path / difficulty, traj, traj_dir, args.cameras)
            ok, detail = validate_trajectory(traj_dir, args.cameras)
            n_done += 1
            if not ok:
                print(f"  WARNING: inconsistent trajectory {traj_name}: {detail}")
                invalid.append((traj_name, detail))

    print(f"Processed {n_done} trajectories into {output_dir}")
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
