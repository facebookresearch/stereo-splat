#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""StereoSplat demo: rectified stereo pair(s) -> 3DGS .ply + rendered novel views.

Usage:
    # single rectified stereo pair
    python scripts/demo.py --left left.png --right right.png \\
        --intrinsics FX FY CX CY --baseline 0.25 \\
        --checkpoint checkpoints/stereosplat_v1.safetensors \\
        --fs-checkpoint-dir checkpoints/foundation_stereo/11-33-40 \\
        --output-dir outputs/demo

    # short stereo sequence: <dir>/left/*.png and <dir>/right/*.png (sorted),
    # plus left-camera camera-to-world poses (metric, OpenCV axes)
    python scripts/demo.py --sequence-dir my_seq --poses my_seq/poses_left.npy \\
        --intrinsics FX FY CX CY --baseline 0.25 \\
        --checkpoint checkpoints/stereosplat_v1.safetensors --output-dir outputs/demo

Intrinsics are in pixels of the input images. Images are resized and
center-cropped to ``--img-size`` (default 320x640, the training resolution)
and the intrinsics are adjusted accordingly.

Outputs in ``--output-dir``:
    gaussians.ply        standard 3D Gaussian Splatting layout
    input_depth_*.png    predicted depth of the input left views (colormap)
    render_rgb_*.png     rendered views (000 = first input camera, then orbit)
    render_depth_*.png   rendered depth (colormap)
    cameras.json         intrinsics and world-to-camera poses of the renders

Rendering uses gsplat and needs a CUDA GPU; ``--no-render`` only exports the
``.ply`` (and runs on CPU, slowly).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch

from stereosplat.inference import (
    build_inference_model,
    find_stereo_sequence,
    load_poses,
    load_rgb,
    orbit_cameras,
    predict_gaussians,
    prepare_stereo_inputs,
    render_views,
    require_cuda,
    scene_target_depth,
)
from stereosplat.utils.ply import gaussians_to_numpy, save_ply


def default_fs_dir() -> str:
    root = os.environ.get("CKPT_ROOT", "checkpoints")
    return os.path.join(root, "foundation_stereo", "11-33-40")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = parser.add_argument_group("input")
    src.add_argument("--left", type=str, help="Left image of a rectified pair.")
    src.add_argument("--right", type=str, help="Right image of a rectified pair.")
    src.add_argument(
        "--sequence-dir",
        type=str,
        help="Directory with left/ and right/ image folders (frames paired in sorted order).",
    )
    src.add_argument(
        "--poses",
        type=str,
        default=None,
        help="Left camera-to-world poses for --sequence-dir: .npy (N, 4, 4) or a "
        "text file with 12/16 values per line. Required for more than one frame.",
    )
    src.add_argument(
        "--stride", type=int, default=1, help="Frame stride for sequences."
    )
    src.add_argument(
        "--max-frames",
        type=int,
        default=4,
        help="Maximum number of stereo pairs used from a sequence.",
    )
    src.add_argument(
        "--intrinsics",
        type=float,
        nargs=4,
        required=True,
        metavar=("FX", "FY", "CX", "CY"),
        help="Pinhole intrinsics of the (rectified) input images, in pixels.",
    )
    src.add_argument(
        "--baseline", type=float, required=True, help="Stereo baseline (m)."
    )
    src.add_argument(
        "--img-size",
        type=int,
        nargs=2,
        default=[320, 640],
        metavar=("H", "W"),
        help="Network input size (multiple of 32).",
    )

    model = parser.add_argument_group("model")
    model.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(
            os.environ.get("CKPT_ROOT", "checkpoints"), "stereosplat_v1.safetensors"
        ),
        help="StereoSplat release or training checkpoint.",
    )
    model.add_argument(
        "--fs-checkpoint-dir",
        type=str,
        default=default_fs_dir(),
        help="FoundationStereo 11-33-40 folder (cfg.yaml + model_best_bp2.pth).",
    )
    model.add_argument(
        "--config",
        type=str,
        default=None,
        help="Training config YAML; only needed for training checkpoints that do "
        "not store their model config.",
    )
    model.add_argument(
        "--device", type=str, default=None, help="Default: cuda if available."
    )

    out = parser.add_argument_group("output")
    out.add_argument("--output-dir", type=str, default="outputs/demo")
    out.add_argument(
        "--ply-sh-degree",
        type=int,
        default=3,
        help="SH degree written to the .ply (higher orders zero-padded; most viewers expect 3).",
    )
    out.add_argument("--no-render", action="store_true", help="Only export the .ply.")
    out.add_argument("--num-views", type=int, default=8, help="Number of orbit views.")
    out.add_argument(
        "--orbit-radius",
        type=float,
        default=None,
        help="Orbit radius in meters (default: the stereo baseline).",
    )
    out.add_argument(
        "--forward",
        type=float,
        default=0.0,
        help="Move the orbit this many meters along the optical axis.",
    )
    args = parser.parse_args()

    if args.sequence_dir is None and (args.left is None or args.right is None):
        parser.error("Give either --left and --right, or --sequence-dir.")
    if args.sequence_dir is not None and (args.left or args.right):
        parser.error("--left/--right and --sequence-dir are mutually exclusive.")
    if not args.no_render:
        try:
            require_cuda("Rendering novel views")
        except RuntimeError as e:
            parser.error(str(e))
    return args


def colorize_depth(depth: np.ndarray, max_value: float | None = None) -> np.ndarray:
    """Depth map -> RGB uint8 (turbo colormap, near = red)."""
    valid = np.isfinite(depth) & (depth > 0)
    if max_value is None:
        max_value = float(np.percentile(depth[valid], 98)) if valid.any() else 1.0
    inv = np.zeros_like(depth, dtype=np.float32)
    inv[valid] = np.clip(depth[valid] / max(max_value, 1e-6), 0.0, 1.0)
    rgb = matplotlib.colormaps["turbo"](1.0 - inv)[..., :3]
    rgb[~valid] = 0.0
    return (rgb * 255.0).astype(np.uint8)


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.sequence_dir is not None:
        paths = find_stereo_sequence(args.sequence_dir)
        poses = load_poses(args.poses) if args.poses else None
        if poses is not None and len(poses) != len(paths):
            raise ValueError(
                f"{args.poses}: {len(poses)} poses for {len(paths)} frames."
            )
        indices = list(range(0, len(paths), args.stride))[: args.max_frames]
        paths = [paths[i] for i in indices]
        poses = poses[indices] if poses is not None else None
    else:
        paths = [(Path(args.left), Path(args.right))]
        poses = None
    pairs = [(load_rgb(left), load_rgb(right)) for left, right in paths]
    inputs = prepare_stereo_inputs(
        pairs,
        intrinsics=args.intrinsics,
        baseline=args.baseline,
        left_c2w=poses,
        img_size=tuple(args.img_size),
        frame_names=[p[0].stem for p in paths],
    )
    print(f"{inputs.num_pairs} stereo pair(s) at {inputs.image_size}, device {device}")

    model = build_inference_model(
        args.checkpoint, args.fs_checkpoint_dir, config_path=args.config, device=device
    )
    gaussians, depths = predict_gaussians(model, inputs, device)

    ply_path = output_dir / "gaussians.ply"
    num = save_ply(
        ply_path, gaussians_to_numpy(gaussians), sh_degree=args.ply_sh_degree
    )
    print(f"Wrote {num} Gaussians to {ply_path}")

    depths_np = depths.cpu().numpy()
    max_vis = float(np.percentile(depths_np[depths_np > 0], 98))
    for i, name in enumerate(inputs.frame_names):
        save_rgb(
            output_dir / f"input_depth_{i:03d}_{name}.png",
            colorize_depth(depths_np[i], max_vis),
        )

    if args.no_render:
        return

    target_depth = scene_target_depth(depths, model.max_depth)
    radius = args.baseline if args.orbit_radius is None else args.orbit_radius
    w2c = orbit_cameras(args.num_views, radius, target_depth, forward=args.forward)
    K = inputs.intrinsics[0, 0]
    renders = render_views(gaussians, w2c, K, inputs.image_size)
    for i in range(w2c.shape[0]):
        rgb = (
            (renders["colors"][i].permute(1, 2, 0).numpy() * 255.0)
            .round()
            .astype(np.uint8)
        )
        save_rgb(output_dir / f"render_rgb_{i:03d}.png", rgb)
        alpha = renders["alphas"][i].numpy()
        depth = renders["depths"][i].numpy() / np.maximum(alpha, 1e-6)
        depth[alpha < 0.1] = 0.0
        save_rgb(
            output_dir / f"render_depth_{i:03d}.png", colorize_depth(depth, max_vis)
        )
    with open(output_dir / "cameras.json", "w") as f:
        json.dump(
            {
                "image_size": list(inputs.image_size),
                "intrinsics": K.tolist(),
                "world_to_camera": w2c.tolist(),
                "convention": "OpenCV axes; world = first input left camera; meters",
            },
            f,
            indent=2,
        )
    print(f"Wrote {w2c.shape[0]} rendered views to {output_dir}")


if __name__ == "__main__":
    main()
