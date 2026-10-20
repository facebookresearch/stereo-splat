# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Model construction and checkpoint loading.

Checkpoint formats accepted by :func:`load_checkpoint`:

* Full training checkpoints written by ``scripts/train.py``:
  ``{"model_state_dict": ..., "optimizer_state_dict": ..., "epoch": ...}``.
  The state dict contains the frozen backbone under ``fs_model.*``.
* Release checkpoints (``version`` key): ``{"version": 1, "config": {...},
  "state_dict": ...}`` holding the StereoSplat head plus the FoundationStereo
  BatchNorm buffers that differ from the public FoundationStereo weights; all
  other ``fs_model.*`` tensors come from the FoundationStereo checkpoint loaded
  by :func:`build_model`.

Key remap: the public module names are identical to the ones used to train the
paper model, so :func:`remap_state_dict` only strips a DistributedDataParallel
``module.`` prefix. See ``KEY_REMAP`` (empty).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Any

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from stereosplat.backbones.foundation_stereo import build_foundation_stereo
from stereosplat.models.gaussian_adapter import GaussianAdapter
from stereosplat.models.stereosplat import StereoSplat
from stereosplat.utils.training import freeze_model

logger: logging.Logger = logging.getLogger(__name__)

MODEL_NAMES: frozenset[str] = frozenset({"stereosplat"})

# Old prefix -> new prefix. Empty: module names are unchanged.
KEY_REMAP: dict[str, str] = {}

# Settings of the released model; other values are not supported by this code.
_FIXED_SETTINGS: dict[str, Any] = {
    "cross_view_mode": "attention_multiscale",
    "use_learned_weights": True,
    "use_residual_gating": False,
}


def _check_config(model_config: Mapping[str, Any]) -> None:
    name = model_config.get("name", "stereosplat")
    if name not in MODEL_NAMES:
        raise ValueError(
            f"Unknown model name {name!r}; expected one of {sorted(MODEL_NAMES)}."
        )
    for key, expected in _FIXED_SETTINGS.items():
        if key in model_config and model_config[key] != expected:
            raise ValueError(
                f"model.{key}={model_config[key]!r} is not supported; the released "
                f"StereoSplat model uses {key}={expected!r}."
            )


def build_foundation_stereo_backbone(
    fs_checkpoint_path: str,
    load_weights: bool = True,
    fs_root: str | None = None,
) -> torch.nn.Module:
    """Build the frozen FoundationStereo backbone.

    Args:
        fs_checkpoint_path: Directory with the FoundationStereo ``cfg.yaml`` and
            ``model_best_bp2.pth`` (the public ViT-S "11-33-40" release).
        load_weights: Load ``model_best_bp2.pth`` (disable only for tests).
        fs_root: FoundationStereo source checkout (see
            ``stereosplat.backbones.foundation_stereo``).
    """
    fs_config = OmegaConf.load(os.path.join(fs_checkpoint_path, "cfg.yaml"))

    # Construct on rank 0 first so torch.hub code downloads happen only once.
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() == 0:
            backbone = build_foundation_stereo(fs_config, root=fs_root)
        dist.barrier()
        if dist.get_rank() != 0:
            backbone = build_foundation_stereo(fs_config, root=fs_root)
    else:
        backbone = build_foundation_stereo(fs_config, root=fs_root)

    if load_weights:
        ckpt_path = os.path.join(fs_checkpoint_path, "model_best_bp2.pth")
        ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
        backbone.load_state_dict(ckpt["model"])

    backbone.eval()
    freeze_model(backbone)
    return backbone


def build_model(
    model_config: Mapping[str, Any],
    backbone: torch.nn.Module | None = None,
    load_backbone_weights: bool = True,
) -> StereoSplat:
    """Build StereoSplat from the ``model`` section of a config.

    Args:
        model_config: Model config (see ``configs/train_stereosplat.yaml``).
        backbone: Optional pre-built backbone (used by tests).
        load_backbone_weights: Load the FoundationStereo weights.
    """
    _check_config(model_config)
    if backbone is None:
        backbone = build_foundation_stereo_backbone(
            model_config["fs_checkpoint_path"],
            load_weights=load_backbone_weights,
            fs_root=model_config.get("fs_root", None),
        )

    gaussian_adapter = GaussianAdapter(
        sh_degree=model_config["sh_degree"],
        gaussian_scale_min=model_config["gaussian_scale_min"],
        gaussian_scale_max=model_config["gaussian_scale_max"],
    )

    return StereoSplat(
        backbone,
        gaussian_adapter,
        max_depth=model_config["max_depth"],
        merge_distance=model_config.get("merge_distance", 0.003),
        appearance_resolution=model_config.get("appearance_resolution", 0.2),
        weight_temperature_init=model_config.get("weight_temperature_init", 1.0),
        raw_gaussian_channels=model_config.get("raw_gaussian_channels", 32),
        cross_view_num_heads=model_config.get("cross_view_num_heads", 4),
        cross_view_num_layers=model_config.get("cross_view_num_layers", 4),
        cross_view_ffn_ratio=model_config.get("cross_view_ffn_ratio", 4),
    )


def remap_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map checkpoint keys onto the current module names."""
    out: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        for old, new in KEY_REMAP.items():
            if key.startswith(old):
                key = new + key[len(old) :]
                break
        out[key] = value
    return out


def load_checkpoint(
    model: StereoSplat,
    checkpoint_path: str,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load StereoSplat weights; returns the checkpoint metadata.

    Full checkpoints are loaded with ``strict=True``. Release checkpoints must
    cover every non-backbone key; missing keys are allowed only under
    ``fs_model.`` (those weights come from the FoundationStereo checkpoint).
    """
    ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if "version" in ckpt:
        state = remap_state_dict(ckpt["state_dict"])
        missing, unexpected = model.load_state_dict(state, strict=False)
        bad_missing = [k for k in missing if not k.startswith("fs_model.")]
        if bad_missing or unexpected:
            raise RuntimeError(
                f"Checkpoint {checkpoint_path} does not match StereoSplat: "
                f"missing={bad_missing[:10]} unexpected={list(unexpected)[:10]}"
            )
    else:
        state = remap_state_dict(ckpt["model_state_dict"])
        model.load_state_dict(state, strict=True)
    logger.info(
        f"Loaded checkpoint {checkpoint_path} (epoch {ckpt.get('epoch', '?')}, "
        f"version {ckpt.get('version', '-')})"
    )
    return {
        k: v
        for k, v in ckpt.items()
        if k not in ("state_dict", "model_state_dict", "optimizer_state_dict")
    }
