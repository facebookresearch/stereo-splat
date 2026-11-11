#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Convert a training checkpoint into a release checkpoint.

The release format (``.safetensors`` or ``.pth``, see
``stereosplat.models.build``) holds the StereoSplat weights and the model
config only; the optimizer state is dropped and the FoundationStereo weights
are never included (they are loaded from NVIDIA's checkpoint).

Usage:
    python scripts/export_weights.py \\
        --checkpoint outputs/stereosplat/checkpoints/best_checkpoint.pt \\
        --output checkpoints/stereosplat_public.safetensors

The exported file can be used wherever the released weights are accepted
(``scripts/evaluate.py --checkpoint``, ``scripts/demo.py``).
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import torch
from omegaconf import OmegaConf

from stereosplat.models.build import (
    BACKBONE_PREFIX,
    read_checkpoint,
    remap_state_dict,
    save_release_checkpoint,
)

# Machine-specific model config keys that are not stored in release files.
LOCAL_KEYS: tuple[str, ...] = ("fs_checkpoint_path", "fs_root")


def release_model_config(
    ckpt: dict[str, Any], config_path: str | None
) -> dict[str, Any]:
    """Model config of a training checkpoint, without machine-specific keys."""
    if config_path is not None:
        cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=False)
        assert isinstance(cfg, dict)
        model_cfg = dict(cfg["model"])
        model_cfg.setdefault("max_depth", cfg.get("max_depth", 80.0))
    elif "model_config" in ckpt:
        model_cfg = dict(ckpt["model_config"])
    else:
        raise ValueError(
            "The checkpoint has no model config; pass the training config YAML "
            "with --config."
        )
    for key in LOCAL_KEYS:
        model_cfg.pop(key, None)
    return model_cfg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, help="Training checkpoint")
    parser.add_argument(
        "--output", required=True, help="Release checkpoint (.safetensors or .pth)"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Training config YAML (only for checkpoints without a stored model "
        "config)",
    )
    args = parser.parse_args()

    ckpt = read_checkpoint(args.checkpoint)
    if "model_state_dict" not in ckpt:
        raise ValueError(f"{args.checkpoint} is not a training checkpoint.")
    state = {
        k: v
        for k, v in remap_state_dict(ckpt["model_state_dict"]).items()
        if not k.startswith(BACKBONE_PREFIX)
    }
    config = release_model_config(ckpt, args.config)
    save_release_checkpoint(args.output, state, config)

    back = read_checkpoint(args.output)
    assert back["config"] == config, (back["config"], config)
    assert back["state_dict"].keys() == state.keys()
    for k, v in state.items():
        assert torch.equal(back["state_dict"][k], v.cpu()), k
    num_params = sum(v.numel() for v in state.values())
    print(
        f"Wrote {args.output}: {len(state)} tensors, {num_params:,} values, "
        f"epoch {ckpt.get('epoch', '?')}, config {config}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
