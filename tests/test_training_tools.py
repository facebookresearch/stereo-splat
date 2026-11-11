# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Backbone freezing, TartanAir --keep-every conversion and weight export."""

from __future__ import annotations

import importlib.util
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import MODEL_CONFIG, StubBackbone
from PIL import Image

from stereosplat.datasets.factory import get_dataset
from stereosplat.inference import load_model_config
from stereosplat.models.build import (
    build_model,
    freeze_model,
    load_checkpoint,
    read_checkpoint,
    trainable_state_dict,
)
from stereosplat.utils.preprocessing import get_transforms

REPO = Path(__file__).resolve().parents[1]


def _load_script(rel_path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stub_model() -> torch.nn.Module:
    cfg = dict(MODEL_CONFIG, cross_view_num_layers=1, max_depth=80.0)
    return build_model(cfg, backbone=freeze_model(StubBackbone()))


def test_freeze_model() -> None:
    net = freeze_model(
        torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2))
    )
    assert not net.training and not any(m.training for m in net.modules())
    assert not any(p.requires_grad for p in net.parameters())


def test_check_backbone_frozen() -> None:
    train = _load_script("scripts/train.py", "train_script")
    model = _stub_model()
    model.train()  # keeps the backbone in eval mode
    train.check_backbone_frozen(model)
    assert any(p.requires_grad for p in model.parameters())

    model.fs_model.train()
    with pytest.raises(RuntimeError, match="train_mode"):
        train.check_backbone_frozen(model)
    model.fs_model.eval()
    model.fs_model.disp.weight.requires_grad_(True)
    with pytest.raises(RuntimeError, match="trainable"):
        train.check_backbone_frozen(model)


# ---------------------------------------------------------------------------
# TartanAir v1 conversion with --keep-every
# ---------------------------------------------------------------------------

ENV, DIFF, TRAJ = "testenv", "Easy", "P000"
NUM_FRAMES = 13
H, W = 24, 32


def _write_raw_tartanair(root: Path) -> None:
    """Raw zips in the official TartanAir v1 layout (one trajectory)."""
    rng = np.random.default_rng(0)
    out = root / ENV / DIFF
    out.mkdir(parents=True)
    prefix = f"{ENV}/{DIFF}/{TRAJ}"
    for side in ("left", "right"):
        with zipfile.ZipFile(out / f"image_{side}.zip", "w") as zf:
            for i in range(NUM_FRAMES):
                buf = io.BytesIO()
                rgb = rng.integers(0, 255, size=(H, W, 3), dtype=np.uint8)
                Image.fromarray(rgb).save(buf, "PNG")
                zf.writestr(f"{prefix}/image_{side}/{i:06d}_{side}.png", buf.getvalue())
            lines = [
                f"{0.1 * i} {0.25 if side == 'right' else 0.0} 0 0 0 0 1"
                for i in range(NUM_FRAMES)
            ]
            zf.writestr(f"{prefix}/pose_{side}.txt", "\n".join(lines) + "\n")
        with zipfile.ZipFile(out / f"depth_{side}.zip", "w") as zf:
            for i in range(NUM_FRAMES):
                buf = io.BytesIO()
                np.save(buf, (2.0 + rng.random((H, W))).astype(np.float32))
                zf.writestr(
                    f"{prefix}/depth_{side}/{i:06d}_{side}_depth.npy", buf.getvalue()
                )


def _dataset(root: Path, stride: int):
    cfg = {
        "dataset_path": str(root),
        "img_size": [H, W],
        "num_views": 2,
        "stride": stride,
        "num_samples": -1,
    }
    return get_dataset(
        "tartan", cfg, get_transforms(), scene_names=[f"{ENV}_{DIFF}_{TRAJ}"]
    )


def test_tartan_keep_every_gives_the_same_windows(tmp_path) -> None:
    convert = _load_script("scripts/prepare_data/tartan.py", "tartan_convert")
    raw = tmp_path / "raw"
    _write_raw_tartanair(raw)
    for keep_every in (1, 5):
        out = tmp_path / f"keep{keep_every}"
        out.mkdir()
        name = convert.process_trajectory(raw / ENV, DIFF, TRAJ, out, keep_every)
        ok, detail = convert.validate_trajectory(out / name)
        assert ok, detail
    n_full = len(
        list((tmp_path / "keep1" / f"{ENV}_{DIFF}_{TRAJ}" / "rgb_left").glob("*.png"))
    )
    n_sub = len(
        list((tmp_path / "keep5" / f"{ENV}_{DIFF}_{TRAJ}" / "rgb_left").glob("*.png"))
    )
    assert (n_full, n_sub) == (NUM_FRAMES, 3)  # frames 0, 5, 10

    full = _dataset(tmp_path / "keep1", stride=5)
    sub = _dataset(tmp_path / "keep5", stride=5)
    assert len(full) == len(sub) > 0
    for i in range(len(full)):
        a, b = full[i], sub[i]
        for key in ("images", "original_images", "depths", "poses", "intrinsics"):
            assert torch.equal(a[key], b[key]), key

    with pytest.raises(ValueError, match="multiple of 5"):
        _dataset(tmp_path / "keep5", stride=1)
    # Re-running with another factor on the same output is refused.
    with pytest.raises(RuntimeError, match="keep-every 5"):
        convert.process_trajectory(raw / ENV, DIFF, TRAJ, tmp_path / "keep5", 1)


# ---------------------------------------------------------------------------
# Export to the release format
# ---------------------------------------------------------------------------


def test_export_weights_roundtrip(tmp_path, monkeypatch) -> None:
    model = _stub_model()
    model_config = dict(MODEL_CONFIG, cross_view_num_layers=1, max_depth=80.0)
    ckpt_path = tmp_path / "best_checkpoint.pt"
    torch.save(
        {
            "epoch": 3,
            "model_state_dict": trainable_state_dict(model),
            "model_config": model_config,
            "optimizer_state_dict": {},
        },
        ckpt_path,
    )
    # Training checkpoints carry their model config.
    cfg = load_model_config(ckpt_path)
    assert "fs_checkpoint_path" not in cfg and cfg["cross_view_num_layers"] == 1

    export = _load_script("scripts/export_weights.py", "export_weights")
    out = tmp_path / "exported.safetensors"
    monkeypatch.setattr(
        sys, "argv", ["export", "--checkpoint", str(ckpt_path), "--output", str(out)]
    )
    assert export.main() == 0

    release = read_checkpoint(out)
    assert release["version"] == 1
    assert "fs_checkpoint_path" not in release["config"]
    assert not any(k.startswith("fs_model.") for k in release["state_dict"])

    reloaded = _stub_model()
    load_checkpoint(reloaded, str(out))
    for k, v in trainable_state_dict(model).items():
        assert torch.equal(reloaded.state_dict()[k], v), k
