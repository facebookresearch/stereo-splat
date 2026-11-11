#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Convert the KITTI Visual Odometry benchmark into the processed layout used by
``stereosplat.datasets.kitti``.

Input (``--input-dir``): the official KITTI odometry downloads
(https://www.cvlibs.net/datasets/kitti/eval_odometry.php), either extracted or
as the original zip files (use ``--extract``)::

    data_odometry_color.zip      -> dataset/sequences/XX/image_2, image_3, times.txt
    data_odometry_calib.zip      -> dataset/sequences/XX/calib.txt, times.txt
    data_odometry_poses.zip      -> dataset/poses/XX.txt   (sequences 00-10)
    data_odometry_velodyne.zip   -> dataset/sequences/XX/velodyne/*.bin
                                    (needed for --mode static only)

Modes:

``static`` (default; the KITTI evaluation set): cuts sequences 00-10 into the
static-scene segments listed in ``STATIC_SEGMENTS`` (start/end times in
seconds, matched against times.txt) and writes one folder per segment, with
sparse LiDAR depth projected into both cameras::

    {seq}_{start}_{end}/              e.g. 00_0m11s_1m00s
        rgb_left/000000.png ...       image_2 (cropped to even width/height)
        rgb_right/000000.png ...      image_3
        depth_left/000000.npy ...     (H, W) float32 sparse LiDAR depth, 0 = no data
        depth_right/000000.npy ...
        poses_left.npy                (N, 4, 4) float32 camera-to-world
        poses_right.npy
        intrinsics_left.npy           (3, 3) float32
        intrinsics_right.npy
        timestamps.npy                (N,) float64 seconds

``full``: one folder per sequence ({seq}/) with the same files but no depth and
no time cropping.

Example::

    python scripts/prepare_data/kitti.py \\
        --input-dir /data/raw/kitti_odometry --extract \\
        --output-dir /data/processed/kitti_static
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

RAW_ZIP_FILES = [
    "data_odometry_calib.zip",
    "data_odometry_color.zip",
    "data_odometry_poses.zip",
    "data_odometry_velodyne.zip",
]

# Static (no independently moving objects) segments per sequence, "m:ss".
STATIC_SEGMENTS: dict[str, list[tuple[str, str]]] = {
    "00": [
        ("0:11", "1:00"),
        ("1:06", "1:17"),
        ("1:22", "1:43"),
        ("1:45", "2:19"),
        ("2:23", "2:58"),
        ("3:08", "3:38"),
        ("3:40", "4:55"),
        ("5:06", "5:56"),
        ("6:33", "7:05"),
        ("7:11", "7:15"),
    ],
    "02": [
        ("0:13", "0:59"),
        ("1:05", "1:24"),
        ("2:12", "2:47"),
        ("2:52", "7:46"),
    ],
    "03": [
        ("0:25", "1:20"),
    ],
    "05": [
        ("0:00", "0:45"),
        ("0:56", "1:11"),
        ("1:56", "2:02"),
        ("2:11", "3:05"),
        ("3:14", "3:24"),
        ("3:30", "3:41"),
        ("3:45", "3:53"),
    ],
    "07": [
        ("0:04", "0:14"),
        ("0:47", "1:02"),
    ],
    "08": [
        ("0:24", "0:47"),
        ("1:21", "1:48"),
        ("3:28", "3:50"),
        ("4:00", "4:42"),
        ("5:25", "5:38"),
        ("5:48", "6:03"),
    ],
    "09": [
        ("0:07", "0:47"),
        ("1:05", "1:43"),
    ],
    "10": [
        ("0:00", "0:52"),
    ],
}


def _time_str_to_seconds(t: str) -> float:
    minutes, seconds = t.strip().split(":")
    return int(minutes) * 60 + int(seconds)


def _time_str_to_label(t: str) -> str:
    minutes, seconds = t.strip().split(":")
    return f"{int(minutes)}m{int(seconds):02d}s"


def segment_folder_name(seq: str, start: str, end: str) -> str:
    return f"{seq}_{_time_str_to_label(start)}_{_time_str_to_label(end)}"


# ---------------------------------------------------------------------------
# Raw KITTI readers
# ---------------------------------------------------------------------------


def extract_zips(input_dir: Path) -> None:
    for zip_name in RAW_ZIP_FILES:
        zip_path = input_dir / zip_name
        marker = input_dir / f".{zip_name}.extracted"
        if not zip_path.exists() or marker.exists():
            continue
        print(f"Extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(path=input_dir)
        marker.touch()


def parse_calib(calib_path: Path) -> dict[str, np.ndarray] | None:
    if not calib_path.exists():
        return None
    calib = {}
    with open(calib_path) as f:
        for line in f:
            if ":" not in line:
                continue
            key, vals = line.split(":", 1)
            calib[key.strip()] = np.array([float(x) for x in vals.split()]).reshape(
                3, 4
            )
    return calib


def load_poses(pose_path: Path) -> list[np.ndarray] | None:
    """KITTI ground-truth poses: cam0-to-world, one flattened 3x4 per line."""
    if not pose_path.exists():
        return None
    poses = []
    with open(pose_path) as f:
        for line in f:
            vals = [float(x) for x in line.strip().split()]
            T = np.eye(4)
            T[:3, :] = np.array(vals).reshape(3, 4)
            poses.append(T)
    return poses


def load_timestamps(times_path: Path) -> list[float] | None:
    if not times_path.exists():
        return None
    with open(times_path) as f:
        return [float(line.strip()) for line in f if line.strip()]


def find_kitti_dirs(input_dir: Path, seq: str) -> tuple[Path, Path] | None:
    """Return (sequence_dir, poses_dir) for ``dataset/...`` or flat layouts."""
    for root in (input_dir / "dataset", input_dir):
        seq_dir = root / "sequences" / seq
        if seq_dir.exists():
            return seq_dir, root / "poses"
    return None


def load_calibration(
    seq_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Return (K, T_cam0_cam2, T_cam0_cam3, raw calib).

    K is taken from P0 (all rectified cameras share it). The color cameras are
    offset from cam0 by a pure translation t = K^-1 P[:, 3]; T_cam0_camX maps
    camX coordinates to cam0 coordinates.
    """
    calib = parse_calib(seq_dir / "calib.txt")
    if calib is None:
        raise FileNotFoundError(f"calib.txt not found in {seq_dir}")
    for key in ("P0", "P2", "P3"):
        if key not in calib:
            raise KeyError(f"{key} missing in {seq_dir / 'calib.txt'}")

    K = calib["P0"][:3, :3].copy().astype(np.float32)
    K_inv = np.linalg.inv(K)
    t2 = K_inv @ calib["P2"][:, 3]
    t3 = K_inv @ calib["P3"][:, 3]
    T_cam0_cam2 = np.eye(4, dtype=np.float32)
    T_cam0_cam2[:3, 3] = -t2
    T_cam0_cam3 = np.eye(4, dtype=np.float32)
    T_cam0_cam3[:3, 3] = -t3
    return K, T_cam0_cam2, T_cam0_cam3, calib


def save_rgb_frames_as_png(image_paths: list[Path], output_dir: Path) -> None:
    """Save images as RGB PNGs, cropping to even width/height."""
    if not image_paths:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, img_path in enumerate(image_paths):
        with Image.open(img_path) as img:
            rgb_img = img.convert("RGB")
            w, h = rgb_img.size
            enc_w, enc_h = w - (w % 2), h - (h % 2)
            if enc_w != w or enc_h != h:
                rgb_img = rgb_img.crop((0, 0, enc_w, enc_h))
            rgb_img.save(output_dir / f"{idx:06d}.png", "PNG")


def save_poses_and_intrinsics(
    output_dir: Path,
    poses_cam0: list[np.ndarray],
    frame_indices: list[int],
    K: np.ndarray,
    T_cam0_cam2: np.ndarray,
    T_cam0_cam3: np.ndarray,
) -> None:
    poses_l, poses_r = [], []
    for i in frame_indices:
        T_w_cam0 = poses_cam0[i].astype(np.float32)
        poses_l.append(T_w_cam0 @ T_cam0_cam2)
        poses_r.append(T_w_cam0 @ T_cam0_cam3)
    for name, arr in [
        ("poses_left.npy", np.stack(poses_l).astype(np.float32)),
        ("poses_right.npy", np.stack(poses_r).astype(np.float32)),
        ("intrinsics_left.npy", K),
        ("intrinsics_right.npy", K),
    ]:
        path = output_dir / name
        if not path.exists():
            np.save(path, arr)


# ---------------------------------------------------------------------------
# LiDAR depth
# ---------------------------------------------------------------------------


def load_velodyne_bin(bin_path: Path) -> np.ndarray:
    """(N, 4) float32 points (x, y, z, reflectance) in the velodyne frame."""
    return np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)


def project_velodyne_to_depth(
    pts_velo: np.ndarray,
    Tr: np.ndarray,
    P_cam: np.ndarray,
    img_h: int,
    img_w: int,
) -> np.ndarray:
    """Project LiDAR points into a sparse (img_h, img_w) depth map (0 = empty).

    Points are moved to the rectified cam0 frame with Tr, projected with P_cam
    (P2 left, P3 right), rounded to the nearest pixel, and the closest point is
    kept per pixel. The stored value is the z coordinate in the cam0 frame.
    """
    pts_hom = pts_velo[:, :3].T
    pts_hom = np.vstack([pts_hom, np.ones((1, pts_hom.shape[1]))])
    pts_cam = Tr @ pts_hom
    pts_cam = pts_cam[:, pts_cam[2, :] > 0]

    pts_cam_hom = np.vstack([pts_cam, np.ones((1, pts_cam.shape[1]))])
    pts_img = P_cam @ pts_cam_hom
    u = np.round(pts_img[0, :] / pts_img[2, :]).astype(np.int32)
    v = np.round(pts_img[1, :] / pts_img[2, :]).astype(np.int32)
    z = pts_cam[2, :]

    valid = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    u, v, z = u[valid], v[valid], z[valid]

    depth = np.zeros((img_h, img_w), dtype=np.float32)
    for ui, vi, zi in zip(u, v, z, strict=True):
        if depth[vi, ui] == 0 or zi < depth[vi, ui]:
            depth[vi, ui] = zi
    return depth


def save_depth_frames(
    frame_indices: list[int],
    velodyne_dir: Path,
    calib: dict[str, np.ndarray],
    img_h: int,
    img_w: int,
    output_dir: Path,
) -> list[str]:
    errors: list[str] = []
    left_dir = output_dir / "depth_left"
    right_dir = output_dir / "depth_right"
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)
    for out_idx, frame_idx in enumerate(frame_indices):
        bin_path = velodyne_dir / f"{frame_idx:06d}.bin"
        if not bin_path.exists():
            errors.append(f"missing velodyne scan {bin_path.name}")
            continue
        pts = load_velodyne_bin(bin_path)
        np.save(
            left_dir / f"{out_idx:06d}.npy",
            project_velodyne_to_depth(pts, calib["Tr"], calib["P2"], img_h, img_w),
        )
        np.save(
            right_dir / f"{out_idx:06d}.npy",
            project_velodyne_to_depth(pts, calib["Tr"], calib["P3"], img_h, img_w),
        )
    return errors


# ---------------------------------------------------------------------------
# Per-folder processing
# ---------------------------------------------------------------------------


def _load_sequence(input_dir: Path, seq: str) -> dict:
    dirs = find_kitti_dirs(input_dir, seq)
    if dirs is None:
        raise FileNotFoundError(f"sequence {seq} not found under {input_dir}")
    seq_dir, poses_dir = dirs
    K, T_cam0_cam2, T_cam0_cam3, calib = load_calibration(seq_dir)
    poses_cam0 = load_poses(poses_dir / f"{seq}.txt")
    if poses_cam0 is None:
        raise FileNotFoundError(f"pose file {poses_dir / f'{seq}.txt'} not found")
    img_dir_l, img_dir_r = seq_dir / "image_2", seq_dir / "image_3"
    if not img_dir_l.exists() or not img_dir_r.exists():
        raise FileNotFoundError(f"image_2/image_3 not found in {seq_dir}")
    return {
        "seq_dir": seq_dir,
        "K": K,
        "T_cam0_cam2": T_cam0_cam2,
        "T_cam0_cam3": T_cam0_cam3,
        "calib": calib,
        "poses_cam0": poses_cam0,
        "timestamps": load_timestamps(seq_dir / "times.txt"),
        "files_l": sorted(img_dir_l.glob("*.png")),
        "files_r": sorted(img_dir_r.glob("*.png")),
    }


def _write_rgb(data: dict, frame_indices: list[int], output_dir: Path) -> None:
    for side, files in (("left", data["files_l"]), ("right", data["files_r"])):
        rgb_dir = output_dir / f"rgb_{side}"
        if not rgb_dir.exists():
            save_rgb_frames_as_png([files[i] for i in frame_indices], rgb_dir)


def process_static_segment(
    input_dir: Path, seq: str, start: str, end: str, output_dir: Path
) -> str:
    """Write one static segment: frames with start <= timestamp <= end."""
    folder = segment_folder_name(seq, start, end)
    data = _load_sequence(input_dir, seq)
    timestamps = data["timestamps"]
    if timestamps is None:
        return f"{folder}: error - times.txt not found"
    total = min(
        len(data["files_l"]),
        len(data["files_r"]),
        len(data["poses_cam0"]),
        len(timestamps),
    )
    start_s, end_s = _time_str_to_seconds(start), _time_str_to_seconds(end)
    indices = [i for i, ts in enumerate(timestamps[:total]) if start_s <= ts <= end_s]
    if not indices:
        return f"{folder}: warning - 0 frames in [{start}, {end}]"

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_rgb(data, indices, output_dir)
    save_poses_and_intrinsics(
        output_dir,
        data["poses_cam0"],
        indices,
        data["K"],
        data["T_cam0_cam2"],
        data["T_cam0_cam3"],
    )
    ts_path = output_dir / "timestamps.npy"
    if not ts_path.exists():
        np.save(ts_path, np.array([timestamps[i] for i in indices], dtype=np.float64))

    errors: list[str] = []
    velodyne_dir = data["seq_dir"] / "velodyne"
    if not velodyne_dir.exists():
        errors.append("velodyne/ not found, depth not written")
    elif "Tr" not in data["calib"]:
        errors.append("Tr missing in calib.txt, depth not written")
    elif not (output_dir / "depth_left").exists():
        with Image.open(data["files_l"][indices[0]]) as img:
            img_w, img_h = img.size
        errors += save_depth_frames(
            indices, velodyne_dir, data["calib"], img_h, img_w, output_dir
        )
    if errors:
        return f"{folder}: partial ({len(indices)} frames): {'; '.join(errors)}"
    return f"{folder}: ok ({len(indices)} frames)"


def process_full_sequence(input_dir: Path, seq: str, output_dir: Path) -> str:
    """Write a whole sequence (no time cropping, no depth)."""
    data = _load_sequence(input_dir, seq)
    n = min(len(data["files_l"]), len(data["files_r"]), len(data["poses_cam0"]))
    if n == 0:
        return f"{seq}: error - no frames"
    indices = list(range(n))
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_rgb(data, indices, output_dir)
    save_poses_and_intrinsics(
        output_dir,
        data["poses_cam0"],
        indices,
        data["K"],
        data["T_cam0_cam2"],
        data["T_cam0_cam3"],
    )
    timestamps = data["timestamps"]
    ts_path = output_dir / "timestamps.npy"
    if timestamps and not ts_path.exists():
        np.save(ts_path, np.array(timestamps[:n], dtype=np.float64))
    return f"{seq}: ok ({n} frames)"


def discover_sequences(input_dir: Path) -> list[str]:
    for root in (input_dir / "dataset", input_dir):
        seq_root = root / "sequences"
        if seq_root.exists():
            return sorted(
                p.name for p in seq_root.iterdir() if p.is_dir() and p.name.isdigit()
            )
    return []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert KITTI odometry to the processed layout."
    )
    parser.add_argument("--input-dir", required=True, help="KITTI odometry root.")
    parser.add_argument("--output-dir", required=True, help="Processed output root.")
    parser.add_argument(
        "--mode",
        choices=["static", "full"],
        default="static",
        help="static: static segments with LiDAR depth (evaluation set); "
        "full: whole sequences without depth.",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=None,
        help="Sequences to process, e.g. 00 02 05 (default: all).",
    )
    parser.add_argument(
        "--split-file",
        default=None,
        help="static mode: only write segments listed in this file "
        "(e.g. data/splits/kitti_static/test.txt).",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract the official data_odometry_*.zip files found in --input-dir "
        "into --input-dir first.",
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=4, help="Parallel worker processes."
    )
    return parser.parse_args()


def _build_tasks(args: argparse.Namespace, input_dir: Path, output_dir: Path) -> list:
    tasks = []
    if args.mode == "full":
        seqs = [s.zfill(2) for s in args.scenes] if args.scenes else None
        for seq in seqs or discover_sequences(input_dir):
            tasks.append((process_full_sequence, (input_dir, seq, output_dir / seq)))
        return tasks

    seqs = [s.zfill(2) for s in args.scenes] if args.scenes else sorted(STATIC_SEGMENTS)
    unknown = [s for s in seqs if s not in STATIC_SEGMENTS]
    if unknown:
        print(f"WARNING: no static segments defined for {unknown}")
    wanted = None
    if args.split_file:
        with open(args.split_file) as f:
            wanted = {ln.strip() for ln in f if ln.strip()}
    for seq in seqs:
        for start, end in STATIC_SEGMENTS.get(seq, []):
            folder = segment_folder_name(seq, start, end)
            if wanted is None or folder in wanted:
                tasks.append(
                    (
                        process_static_segment,
                        (input_dir, seq, start, end, output_dir / folder),
                    )
                )
    return tasks


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.extract:
        extract_zips(input_dir)

    tasks = _build_tasks(args, input_dir, output_dir)
    print(f"KITTI ({args.mode}): {len(tasks)} folders to write into {output_dir}")

    n_bad = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(fn, *fn_args) for fn, fn_args in tasks]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            try:
                msg = f.result()
            except (FileNotFoundError, KeyError) as e:
                msg = f"error - {e}"
            if ": ok" not in msg:
                n_bad += 1
            print(f"  {msg}")
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
