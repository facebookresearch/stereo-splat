# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""torch.distributed helpers for ``torchrun`` launches."""

from __future__ import annotations

import logging
import os

import torch
import torch.distributed as dist

logger: logging.Logger = logging.getLogger("stereosplat")


def log_and_print(msg: str) -> None:
    """Log on the ``stereosplat`` logger (stdout handler configured by scripts)."""
    logger.info(msg)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def setup_distributed() -> tuple[int, int, int, torch.device]:
    """Initialize ``torch.distributed`` from ``torchrun`` environment variables.

    Uses NCCL when CUDA is available and gloo otherwise (CPU runs).

    Returns:
        ``(rank, world_size, local_rank, device)``.
    """
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    elif world_size > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, init_method="env://")
        log_and_print(
            f"Initialized {backend}: rank={rank}, world_size={world_size}, "
            f"local_rank={local_rank}"
        )
    else:
        rank, world_size, local_rank = 0, 1, 0

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    return rank, world_size, local_rank, device


def cleanup_distributed() -> None:
    if dist.is_initialized():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        dist.destroy_process_group()


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def dataloader_workers() -> int:
    """Number of DataLoader workers per rank (2-4, based on CPUs per rank)."""
    num_cpus = os.cpu_count() or 1
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    cpus_per_rank = max(1, num_cpus // local_world)
    return min(4, max(2, cpus_per_rank // 2))
