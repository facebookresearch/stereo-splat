#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Align an externally estimated stereo-VO trajectory (e.g. MAC-VO) to the
ground-truth trajectory of a processed sequence and write the pose files read
by the external-pose evaluation::

    {output_dir}/poses_left.npy                 raw estimate
    {output_dir}/poses_left_aligned.npy         SE(3)-aligned
    {output_dir}/poses_left_aligned_scaled.npy  Sim(3)-aligned
    (+ poses_right*.npy, the left poses shifted by the GT stereo baseline,
    and metrics.json with the ATE before/after alignment)

Use ``--output-dir <poses_dir>/<dataset>/<sequence>`` so that
``scripts/evaluate.py --poses-dir <poses_dir>`` finds the files.

All outputs are (N, 4, 4) float32 camera-to-world matrices in the OpenCV
convention, expressed in the GT world frame. The evaluation only reads the
left file selected by ``--pose-type`` (default ``poses_left_aligned_scaled``)
and derives the right camera from the GT stereo calibration.

Alignment (identical to the procedure used for the paper):
  1. GT left poses are normalized so that frame 0 is the identity.
  2. Umeyama alignment of estimated camera centers to normalized GT centers,
     without scale (SE3) and with scale (Sim3).
  3. The alignment is applied to the full poses
     (R_al = R @ R_est, t_al = s * R @ t_est + t) and mapped back to the GT
     world frame by left-multiplying the first GT pose.

Example::

    python scripts/align_external_poses.py \\
        --est macvo_out/kitti/00_0m11s_1m00s/poses.npy --est-format ned7 \\
        --gt-left /data/processed/kitti_static/00_0m11s_1m00s/poses_left.npy \\
        --gt-right /data/processed/kitti_static/00_0m11s_1m00s/poses_right.npy \\
        --output-dir /data/macvo_poses/kitti/00_0m11s_1m00s
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

# OpenCV/EDN (x right, y down, z forward) -> NED (x forward, y right, z down).
R_EDN2NED = np.array(
    [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)
R_NED2EDN = R_EDN2NED.T


def _to_4x4(R: np.ndarray) -> np.ndarray:
    M = np.eye(4, dtype=R.dtype)
    M[:3, :3] = R
    return M


def poses_ned_to_opencv(poses_ned: np.ndarray) -> np.ndarray:
    """c2w poses with NED world / EDN camera -> OpenCV world and camera."""
    return _to_4x4(R_NED2EDN) @ poses_ned @ _to_4x4(R_EDN2NED)


def pose7_to_matrix(poses7: np.ndarray) -> np.ndarray:
    """(N, 7) [tx ty tz qx qy qz qw] -> (N, 4, 4)."""
    out = np.tile(np.eye(4, dtype=np.float32), (poses7.shape[0], 1, 1))
    out[:, :3, :3] = Rotation.from_quat(poses7[:, 3:7]).as_matrix()
    out[:, :3, 3] = poses7[:, :3]
    return out


def load_estimate(path: str, fmt: str) -> np.ndarray:
    """Load an estimated c2w trajectory and return (N, 4, 4) OpenCV poses."""
    if path.endswith(".npy"):
        est = np.load(path)
    else:
        est = np.loadtxt(path)
    est = est.astype(np.float32)
    if fmt == "cv44":
        return est.reshape(-1, 4, 4)
    if fmt == "ned44":
        return poses_ned_to_opencv(est.reshape(-1, 4, 4))
    if fmt == "ned7":
        return poses_ned_to_opencv(pose7_to_matrix(est.reshape(-1, 7)))
    raise ValueError(f"unknown format {fmt}")


def umeyama_alignment(
    src: np.ndarray, dst: np.ndarray, with_scale: bool = False
) -> tuple[np.ndarray, np.ndarray, float]:
    """Find R, t, s with dst ~= s * R @ src + t for (N, 3) point sets."""
    assert src.shape == dst.shape and src.shape[1] == 3
    n = src.shape[0]
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    sigma_src_sq = np.sum(src_c**2) / n
    cov = (dst_c.T @ src_c) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / sigma_src_sq if with_scale else 1.0
    t = mu_dst - s * R @ mu_src
    return R, t, s


def align_poses(
    poses: np.ndarray, R: np.ndarray, t: np.ndarray, s: float
) -> np.ndarray:
    aligned = np.zeros_like(poses)
    aligned[:, 3, 3] = 1.0
    aligned[:, :3, :3] = R @ poses[:, :3, :3]
    aligned[:, :3, 3] = (R @ poses[:, :3, 3].T).T * s + t[np.newaxis, :]
    return aligned


def ate_rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.linalg.norm(pred - gt, axis=1) ** 2)))


def align_to_gt(
    est_poses: np.ndarray,
    gt_left: np.ndarray,
    gt_right: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Return the six pose arrays and ATE statistics for one sequence."""
    if len(est_poses) != len(gt_left):
        raise ValueError(
            f"estimated trajectory has {len(est_poses)} poses, GT has {len(gt_left)}"
        )
    inv_first = np.linalg.inv(gt_left[0])
    gt_norm = np.stack([inv_first @ p for p in gt_left])
    est_pos = est_poses[:, :3, 3]
    gt_pos = gt_norm[:, :3, 3]

    R_se3, t_se3, s_se3 = umeyama_alignment(est_pos, gt_pos, with_scale=False)
    R_sim3, t_sim3, s_sim3 = umeyama_alignment(est_pos, gt_pos, with_scale=True)

    gt_first = gt_left[0]
    left = {
        "poses_left": gt_first @ est_poses,
        "poses_left_aligned": gt_first @ align_poses(est_poses, R_se3, t_se3, s_se3),
        "poses_left_aligned_scaled": gt_first
        @ align_poses(est_poses, R_sim3, t_sim3, s_sim3),
    }

    baseline = 0.12
    if gt_right is not None:
        baseline = float(np.linalg.norm(gt_left[0][:3, 3] - gt_right[0][:3, 3]))
    stereo_offset = np.eye(4, dtype=np.float32)
    stereo_offset[0, 3] = baseline
    out = dict(left)
    for name, arr in left.items():
        out[name.replace("left", "right")] = arr @ stereo_offset

    stats = {
        "num_frames": len(est_poses),
        "baseline": baseline,
        "sim3_scale": float(s_sim3),
        "ate_unaligned": ate_rmse(est_pos, gt_pos),
        "ate_se3": ate_rmse((R_se3 @ est_pos.T).T + t_se3, gt_pos),
        "ate_sim3": ate_rmse((s_sim3 * (R_sim3 @ est_pos.T)).T + t_sim3, gt_pos),
    }
    return out, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--est", required=True, help="Estimated trajectory file.")
    parser.add_argument(
        "--est-format",
        choices=["ned7", "ned44", "cv44"],
        default="ned7",
        help="ned7: (N, 7) tx ty tz qx qy qz qw, NED world (MAC-VO / TartanAir "
        "convention); ned44: (N, 4, 4) c2w NED; cv44: (N, 4, 4) c2w OpenCV.",
    )
    parser.add_argument("--gt-left", required=True, help="GT left poses (N, 4, 4).")
    parser.add_argument(
        "--gt-right",
        default=None,
        help="GT right poses (N, 4, 4); used for the stereo baseline.",
    )
    parser.add_argument("--output-dir", required=True, help="Output folder.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    est = load_estimate(args.est, args.est_format)
    gt_left = np.load(args.gt_left)
    gt_right = np.load(args.gt_right) if args.gt_right else None
    poses, stats = align_to_gt(est, gt_left, gt_right)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in poses.items():
        np.save(out_dir / f"{name}.npy", arr.astype(np.float32))
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
