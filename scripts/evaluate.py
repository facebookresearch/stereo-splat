#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Evaluate StereoSplat novel-view synthesis with GT and/or external poses.

For every test dataset in the config the model runs in NVS mode (even stereo
pairs of each window are the context, odd pairs are rendered and scored):

* ``gt``  -- ground-truth poses from the dataset;
* ``ext`` -- left-camera poses from ``--poses-dir`` (e.g. MAC-VO, the paper
  protocol), right-camera poses from the GT stereo calibration.

Usage:
    python scripts/evaluate.py --config configs/eval_tartanv2.yaml \
        --checkpoint $CKPT_ROOT/stereosplat_v1.pth \
        --poses-dir $DATA_ROOT/macvo_poses --output-dir outputs/eval_tartanv2

    torchrun --nproc_per_node=8 scripts/evaluate.py ...   # multi-GPU
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import random
import time
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from stereosplat.datasets.factory import check_dataset_not_empty, get_dataset
from stereosplat.evaluation.eval_sampler import EvalSampler
from stereosplat.evaluation.evaluator import RenderingEvaluator
from stereosplat.evaluation.external_poses import (
    align_depth_scale,
    replace_poses_in_batch,
)
from stereosplat.models.build import build_model, load_checkpoint
from stereosplat.rendering.renderer import run_renderer_in_chunk_w_trj_mode
from stereosplat.utils.batch import collate_fn, prepare_input_output_data
from stereosplat.utils.distributed import (
    barrier,
    cleanup_distributed,
    dataloader_workers,
    log_and_print,
    setup_distributed,
    setup_logging,
)
from stereosplat.utils.preprocessing import get_transforms
from stereosplat.utils.visualization import visualize_sample

LOG_EVERY_PERCENTAGE = 2


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _reduce_metrics(
    accumulated: dict[str, float], sample_count: int, device: torch.device
) -> dict[str, float]:
    """Sum metrics and sample counts across ranks and average."""
    keys = sorted(accumulated.keys())
    values = torch.tensor([accumulated[k] for k in keys], device=device)
    count = torch.tensor(sample_count, device=device)
    if dist.is_initialized():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
    if count.item() == 0:
        raise RuntimeError(
            "No samples were evaluated (empty dataset or loader); refusing to "
            "report metrics."
        )
    values = values / count.item()
    return {k: v.item() for k, v in zip(keys, values)}


@torch.no_grad()
def run_nvs_eval(
    model: torch.nn.Module,
    loader: DataLoader,
    evaluator: RenderingEvaluator,
    device: torch.device,
    rank: int,
    max_depth: float,
    writer: SummaryWriter | None,
    dataset_name: str,
    pose_tag: str,
    pose_replace_fn: Callable[[dict[str, Any]], tuple[dict[str, Any], int, int]]
    | None = None,
    scale_align_depth: bool = False,
    profile_runtime: bool = False,
) -> dict[str, float]:
    """NVS evaluation loop shared by the GT-pose and external-pose modes."""
    model.eval()
    raw_model = model.module if isinstance(model, DDP) else model
    accumulated: dict[str, float] = defaultdict(float)
    sample_count = 0
    total_replaced = 0
    total_samples = 0
    inference_time = 0.0
    inference_samples = 0
    render_time = 0.0
    render_frames = 0
    log_interval = max(1, len(loader) * LOG_EVERY_PERCENTAGE // 100)
    num_vis = min(8, len(loader))
    vis_interval = max(1, len(loader) // max(num_vis, 1))
    if profile_runtime and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    for batch_id, data in enumerate(loader):
        if pose_replace_fn is not None:
            data, n_replaced, n_total = pose_replace_fn(data)
            total_replaced += n_replaced
            total_samples += n_total
        else:
            total_samples += data["poses"].shape[0]

        input_data, output_data = prepare_input_output_data(
            data, device, max_depth, eval_nvs=True
        )

        _sync()
        t_start = time.perf_counter()
        output = model(input_data)
        _sync()
        inference_time += time.perf_counter() - t_start
        inference_samples += input_data["images"].shape[0]

        gaussians = raw_model.gaussian_adapter(
            extrinsics=input_data["poses"],
            intrinsics=input_data["intrinsics"],
            depths=output["depths"],
            raw_gaussians=output["raw_gaussians"],
        )
        splats_before = gaussians.means.shape[1]
        gaussians = raw_model.aggregate_gaussians(gaussians)
        splats_after = gaussians.means.shape[1]

        h, w = output_data["images"].shape[-2:]
        if profile_runtime:
            _sync()
        t_render = time.perf_counter()
        renderer_output = run_renderer_in_chunk_w_trj_mode(
            gaussians=gaussians,
            extrinsics=output_data["poses"],
            intrinsics=output_data["intrinsics"],
            image_shape=(h, w),
        )
        if profile_runtime:
            _sync()
            render_time += time.perf_counter() - t_render
            imgs = output_data["images"]
            render_frames += imgs.shape[0] * imgs.shape[1]

        alpha_mask = (renderer_output["alphas"] > 0.1).unsqueeze(2).float()
        output_data["masks"] = output_data["masks"] * alpha_mask

        depth_scale_stats: dict[str, float] = {}
        if scale_align_depth:
            depth_scale_stats = align_depth_scale(renderer_output, output_data)

        metrics = evaluator(renderer_output, output_data)
        metrics.update(depth_scale_stats)
        metrics["splats_before"] = float(splats_before)
        metrics["splats_after"] = float(splats_after)
        metrics["compression_ratio"] = splats_before / max(splats_after, 1)
        metrics["pruned_pct"] = (1.0 - splats_after / max(splats_before, 1)) * 100.0

        batch_size = output_data["images"].shape[0]
        for key, val in metrics.items():
            accumulated[key] += val * batch_size
        sample_count += batch_size

        if rank == 0 and writer is not None and batch_id % vis_interval == 0:
            vis = np.array(visualize_sample(renderer_output, output_data))
            writer.add_image(
                f"test_{pose_tag}/{dataset_name}_NVS/sample_{batch_id}",
                vis,
                0,
                dataformats="HWC",
            )
        if rank == 0 and batch_id % log_interval == 0:
            log_and_print(
                f"Test({pose_tag}) NVS {dataset_name} batch {batch_id}/{len(loader)}: "
                f"PSNR {metrics.get('psnr', 0.0):.4f}, "
                f"pruned {metrics['pruned_pct']:.1f}%, "
                f"compression {metrics['compression_ratio']:.2f}x"
            )

    results = _reduce_metrics(dict(accumulated), sample_count, device)
    if inference_samples > 0 and inference_time > 0:
        results["fps"] = inference_samples / inference_time
    if profile_runtime:
        # Rank-local values.
        results["render_fps"] = render_frames / render_time if render_time > 0 else 0.0
        if torch.cuda.is_available():
            results["peak_mem_gb"] = torch.cuda.max_memory_allocated(device) / (1024**3)

    if rank == 0:
        if pose_replace_fn is not None:
            log_and_print(
                f"  Pose replacement: {total_replaced}/{total_samples} samples used "
                "external poses."
            )
        log_and_print(f"Final Test({pose_tag}) metrics ({dataset_name}/NVS): {results}")
        if writer is not None:
            for key, val in results.items():
                writer.add_scalar(f"test_{pose_tag}/{dataset_name}_NVS/{key}", val)
    return results


def build_test_loaders(
    cfg: dict[str, Any], world_size: int, batch_size: int
) -> dict[str, DataLoader]:
    preprocess = get_transforms()
    num_workers = dataloader_workers()
    loaders: dict[str, DataLoader] = {}
    if not cfg.get("test_datasets"):
        raise ValueError("The config has no test_datasets.")
    allow_missing = bool(cfg.get("allow_missing_scenes", False))
    for name, ds_cfg in cfg["test_datasets"].items():
        log_and_print(f"Loading test dataset: {name}")
        ds_cfg.setdefault("allow_missing_scenes", allow_missing)
        ds = get_dataset(name, ds_cfg, preprocess)
        log_and_print(f"  {check_dataset_not_empty(name, ds, ds_cfg)}")
        loaders[name] = DataLoader(
            ds,
            batch_size=batch_size,
            sampler=EvalSampler(ds) if world_size > 1 else None,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=False,
            prefetch_factor=2 if num_workers > 0 else None,
            multiprocessing_context="spawn" if num_workers > 0 else None,
            collate_fn=collate_fn,
        )
    return loaders


def print_summary(all_metrics: dict[str, dict[str, float]]) -> None:
    log_and_print("=" * 80)
    log_and_print("RESULTS SUMMARY")
    log_and_print("=" * 80)
    for name, m in all_metrics.items():
        log_and_print(
            f"  {name:<28} PSNR={m.get('psnr', 0.0):.4f}  SSIM={m.get('sssim', 0.0):.4f}  "
            f"LPIPS={m.get('lpips', 0.0):.4f}  AbsRel={m.get('absrel', 0.0):.4f}  "
            f"delta1={m.get('delta1', 0.0):.2f}%  DepthL1={m.get('l1', 0.0):.4f}  "
            f"FPS={m.get('fps', 0.0):.2f}"
        )
    log_and_print("=" * 80)


def parse_args() -> tuple[argparse.Namespace, dict[str, Any]]:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=str, required=True, help="YAML config")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--poses-dir",
        type=str,
        default=None,
        help="Root of external poses (<poses_dir>/<dataset>/<seq>/<pose_type>.npy)",
    )
    parser.add_argument("--pose-type", type=str, default="poses_left_aligned_scaled")
    parser.add_argument(
        "--pose-modes",
        type=str,
        default=None,
        help="Comma-separated subset of gt,ext (default: gt,ext with --poses-dir, else gt)",
    )
    parser.add_argument("--num-views", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--scale-align-depth",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Median-scale-align rendered depth before depth metrics",
    )
    parser.add_argument(
        "--profile-runtime",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Measure render-only FPS and peak GPU memory (adds synchronization)",
    )
    parser.add_argument(
        "--allow-missing-scenes",
        action="store_true",
        help="Skip split scenes that are missing on disk (default: fail)",
    )
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(cfg_dict, dict)
    if args.num_views is not None:
        cfg_dict["num_views"] = args.num_views
    if args.batch_size is not None:
        cfg_dict.setdefault("optimization", {})["batch_size"] = args.batch_size
    if args.output_dir is not None:
        cfg_dict.setdefault("logging", {})["output_dir"] = args.output_dir
    if args.scale_align_depth is not None:
        cfg_dict["scale_align_depth"] = args.scale_align_depth
    if args.profile_runtime is not None:
        cfg_dict["profile_runtime"] = args.profile_runtime
    if args.allow_missing_scenes:
        cfg_dict["allow_missing_scenes"] = True
    return args, cfg_dict


def main() -> None:
    setup_logging()
    args, cfg = parse_args()
    rank, world_size, _local_rank, device = setup_distributed()

    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    modes = (
        args.pose_modes.split(",")
        if args.pose_modes
        else (["gt", "ext"] if args.poses_dir else ["gt"])
    )
    if "ext" in modes and not args.poses_dir:
        raise ValueError("--pose-modes ext requires --poses-dir")

    max_depth = cfg["max_depth"]
    cfg["model"]["max_depth"] = max_depth
    num_views = cfg.get("num_views", 3)
    batch_size = (cfg.get("optimization") or {}).get("batch_size", 1)
    for name in cfg.get("test_datasets") or {}:
        cfg["test_datasets"][name]["num_views"] = num_views

    loaders = build_test_loaders(cfg, world_size, batch_size)
    model = build_model(cfg["model"]).to(device)
    load_checkpoint(model, args.checkpoint, map_location=device)
    if world_size > 1:
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
    evaluator = RenderingEvaluator(device=device)

    output_dir = (cfg.get("logging") or {}).get("output_dir")
    writer = None
    if rank == 0 and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(output_dir, "tensorboard"))
    barrier()

    common = dict(
        evaluator=evaluator,
        device=device,
        rank=rank,
        max_depth=max_depth,
        writer=writer,
        scale_align_depth=bool(cfg.get("scale_align_depth", False)),
        profile_runtime=bool(cfg.get("profile_runtime", False)),
    )
    all_metrics: dict[str, dict[str, float]] = {}
    for name, loader in loaders.items():
        if "gt" in modes:
            log_and_print(f"Testing on {name} (NVS) with GT poses...")
            all_metrics[f"{name}_NVS_gt"] = run_nvs_eval(
                model, loader, dataset_name=name, pose_tag="gt", **common
            )
        if "ext" in modes:
            log_and_print(f"Testing on {name} (NVS) with external poses...")
            replace = functools.partial(
                replace_poses_in_batch,
                poses_dir=args.poses_dir,
                dataset_name=name,
                pose_type=args.pose_type,
            )
            all_metrics[f"{name}_NVS_ext"] = run_nvs_eval(
                model,
                loader,
                dataset_name=name,
                pose_tag="ext",
                pose_replace_fn=replace,
                **common,
            )

    if rank == 0:
        print_summary(all_metrics)
        if output_dir:
            path = os.path.join(output_dir, "metrics.json")
            with open(path, "w") as f:
                json.dump(all_metrics, f, indent=2)
            log_and_print(f"Saved metrics to {path}")

    if writer is not None:
        writer.close()
    cleanup_distributed()


if __name__ == "__main__":
    main()
