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

    # resume an interrupted run (or set ``resume_from`` in the config)
    python scripts/train.py --config configs/train_stereosplat.yaml \
        --resume outputs/stereosplat/checkpoints/last_checkpoint.pt

Paths in the config use ``${oc.env:DATA_ROOT}`` / ``${oc.env:CKPT_ROOT}``;
export those variables before launching.

Checkpoints (``<output_dir>/checkpoints``): ``last_checkpoint.pt`` is written
after every epoch and ``best_checkpoint.pt`` whenever the validation NVS PSNR
improves. Both hold the model, optimizer and LR-scheduler state, the epoch,
the global step, the best metric and the RNG states (torch, CUDA, numpy,
python), so training resumes at the next epoch with the same LR schedule.
"""

from __future__ import annotations

import argparse
import contextlib
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


def get_rng_state() -> dict[str, Any]:
    """RNG states of torch (CPU and CUDA), numpy and python.

    Stored as tensors / plain Python types so that the checkpoint can be
    loaded with ``torch.load(weights_only=True)``.
    """
    np_state = np.random.get_state()
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy": {
            "bit_generator": np_state[0],
            "state": torch.from_numpy(np_state[1].astype(np.int64)),
            "pos": int(np_state[2]),
            "has_gauss": int(np_state[3]),
            "cached_gaussian": float(np_state[4]),
        },
        "python": random.getstate(),
    }


def set_rng_state(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"]:
        if len(state["cuda"]) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(state["cuda"])
        else:
            log_and_print(
                f"Checkpoint has {len(state['cuda'])} CUDA RNG states but "
                f"{torch.cuda.device_count()} devices are visible; CUDA RNG "
                "states are not restored."
            )
    np_state = state["numpy"]
    np.random.set_state(
        (
            np_state["bit_generator"],
            np_state["state"].numpy().astype(np.uint32),
            np_state["pos"],
            np_state["has_gauss"],
            np_state["cached_gaussian"],
        )
    )
    version, internal, gauss = state["python"]
    random.setstate((version, tuple(internal), gauss))


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    global_step: int,
    nvs_psnr: float,
    best_nvs_psnr: float,
    checkpoint_dir: str,
    rank: int,
    names: list[str],
) -> None:
    """Write the training state to ``{checkpoint_dir}/{name}`` on rank 0.

    ``epoch`` is the last completed epoch (0-based); ``global_step`` counts
    the training batches processed so far. RNG states are those of rank 0.
    """
    if rank != 0:
        return
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": unwrap(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "nvs_psnr": nvs_psnr,
        "best_nvs_psnr": best_nvs_psnr,
        "rng_state": get_rng_state(),
    }
    for name in names:
        path = os.path.join(checkpoint_dir, name)
        tmp_path = f"{path}.tmp"
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, path)
        log_and_print(
            f"Saved checkpoint (epoch {epoch}, step {global_step}, "
            f"NVS PSNR {nvs_psnr:.4f}) to {path}"
        )


def resume_training_state(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> dict[str, Any]:
    """Restore a checkpoint written by :func:`save_checkpoint`.

    Returns:
        ``epoch`` (last completed epoch), ``global_step``, ``nvs_psnr`` and
        ``best_nvs_psnr``.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    missing = [
        k
        for k in ("scheduler_state_dict", "optimizer_state_dict", "rng_state")
        if k not in ckpt
    ]
    if missing:
        raise ValueError(
            f"{path} is not a resumable training checkpoint (missing {missing}). "
            "Use init_checkpoint_path to initialize the weights only."
        )
    unwrap(model).load_state_dict(ckpt["model_state_dict"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    set_rng_state(ckpt["rng_state"])
    log_and_print(
        f"Resumed from {path}: completed epoch {ckpt['epoch']}, global step "
        f"{ckpt['global_step']}, best NVS PSNR {ckpt['best_nvs_psnr']:.4f}, "
        f"lr {optimizer.param_groups[0]['lr']:.6g}"
    )
    return {
        "epoch": ckpt["epoch"],
        "global_step": ckpt["global_step"],
        "nvs_psnr": ckpt["nvs_psnr"],
        "best_nvs_psnr": ckpt["best_nvs_psnr"],
    }


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


def resolve_dataset_configs(config: dict[str, Any]) -> None:
    """Drop disabled (``null``) dataset entries and set per-dataset defaults.

    A dataset set to ``null`` (e.g. the override ``test_datasets.kitti=null``)
    is skipped. ``num_views`` and ``allow_missing_scenes`` come from the
    top-level config (``allow_missing_scenes`` only if not set per dataset).
    """
    num_views = config["num_views"]
    allow_missing = bool(config.get("allow_missing_scenes", False))
    for ds_type in ["training_datasets", "test_datasets"]:
        entries = config.get(ds_type) or {}
        config[ds_type] = {k: v for k, v in entries.items() if v is not None}
        for ds_cfg in config[ds_type].values():
            ds_cfg["num_views"] = num_views
            ds_cfg.setdefault("allow_missing_scenes", allow_missing)


def create_dataloaders(
    config: dict[str, Any], rank: int, world_size: int
) -> tuple[DataLoader, DataLoader, dict[str, DataLoader], DistributedSampler | None]:
    """Train / validation (10% random split of train) / test dataloaders."""
    resolve_dataset_configs(config)
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
    grad_accum_steps: int = 1,
    vis_every_n_steps: int | None = None,
) -> None:
    """Train for one epoch (in-view synthesis: all views are input and target).

    Losses are logged every ``log_every_n_steps`` batches and the comparison
    image every ``vis_every_n_steps`` batches (default: same as the losses).

    With ``grad_accum_steps > 1`` gradients of that many consecutive batches
    are averaged before each optimizer step (the last step of an epoch may
    use fewer batches).
    """
    model.train()  # StereoSplat.train() keeps the frozen backbone in eval mode.
    accum = max(1, int(grad_accum_steps))
    vis_every = vis_every_n_steps or log_every_n_steps
    num_batches = len(dataloader)

    for batch_idx, batch in enumerate(dataloader):
        global_step = epoch * num_batches + batch_idx
        data = prepare_model_inputs(batch, device, max_depth)

        renderer_output, _, _ = render_batch(model, data, data)
        losses = loss_fn(renderer_output, data)
        total_loss = losses["total_loss"]

        step_now = (batch_idx + 1) % accum == 0 or batch_idx + 1 == num_batches
        if accum == 1 or batch_idx % accum == 0:
            optimizer.zero_grad()
        no_sync = isinstance(model, DDP) and not step_now
        with model.no_sync() if no_sync else contextlib.nullcontext():
            (total_loss / accum if accum > 1 else total_loss).backward()
        if step_now:
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
            log_and_print(
                f"[Epoch {epoch}] [Rank {rank}/{world_size}] "
                f"[Batch {batch_idx}/{len(dataloader)}] "
                f"Loss: {values['total_loss']:.4f} "
                f"(color: {values['color_loss']:.4f}, depth: {values['depth_loss']:.4f}, "
                f"l1: {values['color_l1_loss']:.4f}, ssim: {values['color_ssim_loss']:.4f}, "
                f"lpips: {values['color_lpips_loss']:.4f}) "
                f"lr: {optimizer.param_groups[0]['lr']:.6g}"
            )
        if rank == 0 and writer is not None and batch_idx % vis_every == 0:
            vis = np.array(visualize_sample(renderer_output, data))
            writer.add_image("train/comparison", vis, global_step, dataformats="HWC")

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
    """AdamW with a per-epoch LR schedule (``scheduler.step()`` once per epoch).

    Epochs ``e < warmup_epochs`` use ``min_lr + (lr - min_lr) * e / warmup_epochs``
    (so epoch 0 runs at ``min_lr``); afterwards the LR follows a cosine decay
    from ``lr`` to ``min_lr`` at ``epochs``.
    """
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
        "--resume",
        type=str,
        default=None,
        help="Training checkpoint to resume from (overrides resume_from)",
    )
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
    if args.resume is not None:
        cfg_dict["resume_from"] = args.resume
    return cfg_dict


def main() -> None:
    setup_logging()
    train(parse_config())


def train(cfg: dict[str, Any]) -> None:
    """Run training (and the final test evaluation) for a resolved config."""
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
    global_step = 0
    start_epoch = 0
    resume_from = cfg.get("resume_from")
    if resume_from:
        state = resume_training_state(resume_from, model, optimizer, scheduler)
        start_epoch = state["epoch"] + 1
        global_step = state["global_step"]
        best_nvs_psnr = state["best_nvs_psnr"]
        last_nvs_psnr = state["nvs_psnr"]
        barrier()

    for epoch in range(start_epoch, epochs):
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
            grad_accum_steps=opt_cfg.get("grad_accum_steps", 1),
            vis_every_n_steps=opt_cfg.get("vis_every_n_steps"),
        )
        scheduler.step()
        global_step += len(train_loader)

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
            improved = last_nvs_psnr > best_nvs_psnr
            if improved:
                best_nvs_psnr = last_nvs_psnr
            elif rank == 0:
                log_and_print(
                    f"NVS PSNR {last_nvs_psnr:.4f} did not improve from best {best_nvs_psnr:.4f}"
                )
        else:
            improved = False

        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch,
            global_step,
            last_nvs_psnr,
            best_nvs_psnr,
            checkpoint_dir,
            rank,
            ["best_checkpoint.pt", "last_checkpoint.pt"]
            if improved
            else ["last_checkpoint.pt"],
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
