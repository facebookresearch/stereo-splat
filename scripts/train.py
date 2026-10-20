#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Train StereoSplat (single GPU or multi-GPU / multi-node with torchrun).

Usage:
    # single process
    python scripts/train.py --config configs/train_stereosplat.yaml

    # 8 GPUs on one node
    torchrun --nproc_per_node=8 scripts/train.py --config configs/train_stereosplat.yaml

Paths in the config use ``${oc.env:DATA_ROOT}`` / ``${oc.env:CKPT_ROOT}``;
export those variables before launching.
"""

from __future__ import annotations

import argparse
import math
import os
import random
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from stereosplat.datasets.factory import check_dataset_not_empty, get_dataset
from stereosplat.evaluation.evaluator import RenderingEvaluator
from stereosplat.losses.factory import get_loss
from stereosplat.models.build import build_model, load_checkpoint
from stereosplat.rendering.renderer import run_renderer_in_chunk_w_trj_mode
from stereosplat.utils import training as training_utils
from stereosplat.utils.batch import (
    collate_fn,
    prepare_input_output_data,
    prepare_model_inputs,
)
from stereosplat.utils.distributed import (
    barrier,
    cleanup_distributed,
    dataloader_workers,
    log_and_print,
    setup_distributed,
    setup_logging,
)
from stereosplat.utils.visualization import visualize_sample

LOG_EVERY_PERCENTAGE = 2


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    nvs_psnr: float,
    checkpoint_dir: str,
    rank: int,
    name: str,
) -> None:
    """Write ``{checkpoint_dir}/{name}`` on rank 0."""
    if rank != 0:
        return
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": unwrap(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "nvs_psnr": nvs_psnr,
    }
    path = os.path.join(checkpoint_dir, name)
    torch.save(checkpoint, path)
    log_and_print(
        f"Saved checkpoint (epoch {epoch}, NVS PSNR {nvs_psnr:.4f}) to {path}"
    )


def _make_loader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    sampler: DistributedSampler | None,
    shuffle: bool,
    drop_last: bool = False,
) -> DataLoader:
    num_workers = dataloader_workers()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        prefetch_factor=2 if num_workers > 0 else None,
        multiprocessing_context="spawn" if num_workers > 0 else None,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )


def create_dataloaders(
    config: dict[str, Any], rank: int, world_size: int
) -> tuple[DataLoader, DataLoader, dict[str, DataLoader], DistributedSampler | None]:
    """Train / validation (10% random split of train) / test dataloaders."""
    num_views = config["num_views"]
    allow_missing = bool(config.get("allow_missing_scenes", False))
    for ds_type in ["training_datasets", "test_datasets"]:
        for ds_name in config.get(ds_type) or {}:
            config[ds_type][ds_name]["num_views"] = num_views
            config[ds_type][ds_name].setdefault("allow_missing_scenes", allow_missing)
    if not config.get("training_datasets"):
        raise ValueError("The config has no training_datasets.")

    preprocess = training_utils.get_transforms()
    batch_size = config["optimization"]["batch_size"]

    train_datasets = []
    for name, ds_cfg in config["training_datasets"].items():
        log_and_print(f"Loading training dataset: {name}")
        ds = get_dataset(name, ds_cfg, preprocess)
        log_and_print(f"  {check_dataset_not_empty(name, ds, ds_cfg)}")
        train_datasets.append(ds)
    full_train_data = torch.utils.data.ConcatDataset(train_datasets)

    seed = config.get("seed", 42)
    total = len(full_train_data)
    if total < 2:
        raise ValueError(
            f"Need at least 2 training windows for the train/val split, got {total}."
        )
    val_size = max(1, int(total * 0.1))
    generator = torch.Generator().manual_seed(seed)
    train_data, val_data = torch.utils.data.random_split(
        full_train_data, [total - val_size, val_size], generator=generator
    )

    distributed = world_size > 1
    train_sampler = (
        DistributedSampler(train_data, shuffle=True) if distributed else None
    )
    val_sampler = DistributedSampler(val_data, shuffle=False) if distributed else None
    train_loader = _make_loader(train_data, batch_size, train_sampler, True, True)
    val_loader = _make_loader(val_data, batch_size, val_sampler, False)

    test_loaders = {}
    for name, ds_cfg in (config.get("test_datasets") or {}).items():
        log_and_print(f"Loading test dataset: {name}")
        ds = get_dataset(name, ds_cfg, preprocess)
        log_and_print(f"  {check_dataset_not_empty(name, ds, ds_cfg)}")
        sampler = DistributedSampler(ds, shuffle=False) if distributed else None
        test_loaders[name] = _make_loader(ds, batch_size, sampler, False)

    if len(train_loader) == 0:
        raise ValueError(
            f"The training loader is empty: {len(train_data)} training windows, "
            f"batch_size {batch_size}, world_size {world_size} (drop_last=True)."
        )
    if rank == 0:
        test_info = ", ".join(f"{k}: {len(v.dataset)}" for k, v in test_loaders.items())
        log_and_print(
            f"Total samples - Train: {len(train_data)}, Val: {len(val_data)} "
            f"(10% split), Test: {{{test_info}}}"
        )
    return train_loader, val_loader, test_loaders, train_sampler


def create_model(
    config: dict[str, Any], device: torch.device, rank: int, world_size: int
) -> torch.nn.Module:
    """Build StereoSplat, optionally warm-start it, and wrap it in DDP."""
    model = build_model(config["model"]).to(device)

    init_checkpoint_path = config.get("init_checkpoint_path", None)
    if init_checkpoint_path:
        log_and_print(f"Warm-starting model weights from: {init_checkpoint_path}")
        load_checkpoint(model, init_checkpoint_path, map_location=device)

    if world_size > 1:
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)

    if rank == 0:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log_and_print(f"Model parameters - Total: {total:,}, Trainable: {trainable:,}")
    return model


def render_batch(
    model: torch.nn.Module,
    input_data: dict[str, Any],
    output_data: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], int, int]:
    """Forward, lift, aggregate and render the target views.

    Returns:
        ``(renderer_output, splats_before, splats_after)``.
    """
    outputs = model(input_data)
    raw_model = unwrap(model)
    gaussians = raw_model.gaussian_adapter(
        extrinsics=input_data["poses"],
        intrinsics=input_data["intrinsics"],
        depths=outputs["depths"],
        raw_gaussians=outputs["raw_gaussians"],
    )
    splats_before = gaussians.means.shape[1]
    gaussians = raw_model.aggregate_gaussians(gaussians)
    splats_after = gaussians.means.shape[1]
    h, w = output_data["images"].shape[-2:]
    renderer_output = run_renderer_in_chunk_w_trj_mode(
        gaussians=gaussians,
        extrinsics=output_data["poses"],
        intrinsics=output_data["intrinsics"],
        image_shape=(h, w),
    )
    return renderer_output, splats_before, splats_after


def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device,
    rank: int,
    world_size: int,
    epoch: int,
    writer: SummaryWriter | None,
    max_depth: float,
    log_every_n_steps: int = 100,
    grad_clip: float = 0.0,
) -> None:
    """Train for one epoch (in-view synthesis: all views are input and target)."""
    model.train()  # StereoSplat.train() keeps the frozen backbone in eval mode.

    for batch_idx, batch in enumerate(dataloader):
        global_step = epoch * len(dataloader) + batch_idx
        data = prepare_model_inputs(batch, device, max_depth)

        renderer_output, _, _ = render_batch(model, data, data)
        losses = loss_fn(renderer_output, data)
        total_loss = losses["total_loss"]

        optimizer.zero_grad()
        total_loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if batch_idx % log_every_n_steps == 0:
            values = {k: v.item() for k, v in losses.items()}
            if rank == 0 and writer is not None:
                for k, v in values.items():
                    writer.add_scalar(f"train/{k}", v, global_step)
                writer.add_scalar(
                    "train/lr", optimizer.param_groups[0]["lr"], global_step
                )
                vis = np.array(visualize_sample(renderer_output, data))
                writer.add_image(
                    "train/comparison", vis, global_step, dataformats="HWC"
                )
            log_and_print(
                f"[Epoch {epoch}] [Rank {rank}/{world_size}] "
                f"[Batch {batch_idx}/{len(dataloader)}] "
                f"Loss: {values['total_loss']:.4f} "
                f"(color: {values['color_loss']:.4f}, depth: {values['depth_loss']:.4f}, "
                f"l1: {values['color_l1_loss']:.4f}, ssim: {values['color_ssim_loss']:.4f}, "
                f"lpips: {values['color_lpips_loss']:.4f})"
            )

    barrier()


def _accumulate(acc: dict[str, float], metrics: dict[str, float], n: int) -> None:
    for key, val in metrics.items():
        acc[key] = acc.get(key, 0.0) + val * n


def _reduce(
    acc: dict[str, float], count: int, device: torch.device
) -> dict[str, float]:
    keys = sorted(acc.keys())
    values = torch.tensor([acc[k] for k in keys], device=device, dtype=torch.float64)
    total = torch.tensor(count, device=device, dtype=torch.float64)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(total, op=torch.distributed.ReduceOp.SUM)
    if total.item() > 0:
        values = values / total
    return {k: v.item() for k, v in zip(keys, values)}


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    evaluator: RenderingEvaluator,
    device: torch.device,
    rank: int,
    max_depth: float,
    writer: SummaryWriter | None,
    eval_nvs: bool,
    tag: str,
    step: int,
    reduce_across_ranks: bool,
) -> dict[str, float]:
    """Validation / test loop (IVS or NVS)."""
    model.eval()
    acc: dict[str, float] = {}
    count = 0
    num_vis = min(8, len(loader))
    vis_interval = max(1, len(loader) // max(num_vis, 1))
    log_interval = max(1, len(loader) * LOG_EVERY_PERCENTAGE // 100)

    for batch_idx, batch in enumerate(loader):
        input_data, output_data = prepare_input_output_data(
            batch, device, max_depth, eval_nvs
        )
        renderer_output, before, after = render_batch(model, input_data, output_data)
        if eval_nvs:
            alpha_mask = (renderer_output["alphas"] > 0.1).unsqueeze(2).float()
            output_data["masks"] = output_data["masks"] * alpha_mask

        metrics = evaluator(renderer_output, output_data)
        metrics["splats_before"] = float(before)
        metrics["splats_after"] = float(after)
        metrics["compression_ratio"] = before / max(after, 1)
        metrics["pruned_pct"] = (1.0 - after / max(before, 1)) * 100.0
        n = output_data["images"].shape[0]
        _accumulate(acc, metrics, n)
        count += n

        if rank == 0 and writer is not None and batch_idx % vis_interval == 0:
            vis = np.array(visualize_sample(renderer_output, output_data))
            writer.add_image(f"{tag}/sample_{batch_idx}", vis, step, dataformats="HWC")
        if rank == 0 and batch_idx % log_interval == 0:
            log_and_print(
                f"[{tag}] Batch {batch_idx}/{len(loader)} "
                f"PSNR: {metrics.get('psnr', 0.0):.4f}, SSIM: {metrics.get('sssim', 0.0):.4f}, "
                f"LPIPS: {metrics.get('lpips', 0.0):.4f}, Depth L1: {metrics.get('l1', 0.0):.4f}"
            )

    if reduce_across_ranks:
        avg = _reduce(acc, count, device)
    else:
        avg = {k: v / count for k, v in acc.items()} if count > 0 else {}
    if rank == 0:
        log_and_print(f"[{tag}] Averages: {avg}")
        if writer is not None:
            for key, val in avg.items():
                writer.add_scalar(f"{tag}/{key}", val, step)
    barrier()
    return avg


def build_optimizer_and_scheduler(
    model: torch.nn.Module, opt_cfg: dict[str, Any], epochs: int
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    """AdamW + per-epoch linear warmup (from min_lr) and cosine decay to min_lr."""
    lr = opt_cfg["lr"]
    min_lr = opt_cfg["min_lr"]
    warmup_epochs = opt_cfg["warmup_epochs"]
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, foreach=torch.cuda.is_available())

    def lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return min_lr / lr + (1.0 - min_lr / lr) * epoch / warmup_epochs
        decay_epochs = max(epochs - warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / decay_epochs
        return min_lr / lr + 0.5 * (1.0 - min_lr / lr) * (
            1.0 + math.cos(math.pi * progress)
        )

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def parse_config() -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=str, required=True, help="YAML config")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-views", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--allow-missing-scenes",
        action="store_true",
        help="Skip split scenes that are missing on disk (default: fail)",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf dotlist overrides, e.g. optimization.epochs=1",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(cfg_dict, dict)
    if args.batch_size is not None:
        cfg_dict["optimization"]["batch_size"] = args.batch_size
    if args.num_views is not None:
        cfg_dict["num_views"] = args.num_views
    if args.output_dir is not None:
        cfg_dict.setdefault("logging", {})["output_dir"] = args.output_dir
    if args.allow_missing_scenes:
        cfg_dict["allow_missing_scenes"] = True
    return cfg_dict


def main() -> None:
    setup_logging()
    cfg = parse_config()
    rank, world_size, _local_rank, device = setup_distributed()
    log_and_print(f"Rank {rank}/{world_size} on {device}")

    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if rank == 0:
        log_and_print(f"Configuration:\n{OmegaConf.to_yaml(OmegaConf.create(cfg))}")

    opt_cfg = cfg["optimization"]
    max_depth = cfg["max_depth"]
    cfg["model"]["max_depth"] = max_depth
    epochs = opt_cfg["epochs"]
    validate_every = opt_cfg.get("validate_every", 1)

    train_loader, val_loader, test_loaders, train_sampler = create_dataloaders(
        cfg, rank, world_size
    )
    model = create_model(cfg, device, rank, world_size)
    evaluator = RenderingEvaluator(device=device)
    loss_fn = get_loss(cfg["loss"]).to(device)
    optimizer, scheduler = build_optimizer_and_scheduler(model, opt_cfg, epochs)

    output_dir = (cfg.get("logging") or {}).get("output_dir") or "outputs/stereosplat"
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    writer = None
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(output_dir, "tensorboard"))
        with open(os.path.join(output_dir, "config.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(OmegaConf.create(cfg)))
        log_and_print(f"Output directory: {output_dir}")

    best_nvs_psnr = 0.0
    last_nvs_psnr = 0.0
    for epoch in range(epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        log_and_print(f"Starting epoch {epoch + 1}/{epochs}")
        train_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            rank,
            world_size,
            epoch,
            writer,
            max_depth,
            log_every_n_steps=opt_cfg.get("log_every_n_steps", 100),
            grad_clip=opt_cfg.get("grad_clip", 0.0),
        )
        scheduler.step()

        if (epoch + 1) % validate_every == 0:
            # Validation metrics are rank-local averages (as in the paper runs);
            # only rank 0's value selects the best checkpoint.
            for eval_nvs in (False, True):
                metrics = evaluate_loader(
                    model,
                    val_loader,
                    evaluator,
                    device,
                    rank,
                    max_depth,
                    writer,
                    eval_nvs=eval_nvs,
                    tag=f"val_{'NVS' if eval_nvs else 'IVS'}",
                    step=epoch,
                    reduce_across_ranks=False,
                )
            last_nvs_psnr = metrics.get("psnr", 0.0)
            if last_nvs_psnr > best_nvs_psnr:
                best_nvs_psnr = last_nvs_psnr
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    last_nvs_psnr,
                    checkpoint_dir,
                    rank,
                    "best_checkpoint.pt",
                )
            elif rank == 0:
                log_and_print(
                    f"NVS PSNR {last_nvs_psnr:.4f} did not improve from best {best_nvs_psnr:.4f}"
                )

    save_checkpoint(
        model,
        optimizer,
        epochs - 1,
        last_nvs_psnr,
        checkpoint_dir,
        rank,
        "last_checkpoint.pt",
    )
    barrier()

    for name, loader in test_loaders.items():
        for eval_nvs in (False, True):
            evaluate_loader(
                model,
                loader,
                evaluator,
                device,
                rank,
                max_depth,
                writer,
                eval_nvs=eval_nvs,
                tag=f"test/{name}_{'NVS' if eval_nvs else 'IVS'}",
                step=0,
                reduce_across_ranks=True,
            )

    if writer is not None:
        writer.close()
    cleanup_distributed()
    log_and_print("Training complete!")


if __name__ == "__main__":
    main()
