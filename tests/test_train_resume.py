# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""scripts/train.py save/resume on CPU (stub backbone, fake renderer)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import MODEL_CONFIG, StubBackbone
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
NUM_FRAMES = 8
H, W = 32, 64


def _load_train_script():
    spec = importlib.util.spec_from_file_location(
        "train_script", REPO / "scripts" / "train.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_scene(root: Path) -> None:
    """Tiny TartanAir-v2-style scene: forward motion, 0.25 m stereo baseline."""
    rng = np.random.default_rng(0)
    for side, x_offset in (("lcam_front", 0.0), ("rcam_front", 0.25)):
        (root / f"rgb_{side}").mkdir(parents=True)
        (root / f"depth_{side}").mkdir()
        poses = np.tile(np.eye(4, dtype=np.float32), (NUM_FRAMES, 1, 1))
        poses[:, 0, 3] = x_offset
        poses[:, 2, 3] = 0.1 * np.arange(NUM_FRAMES)
        np.save(root / f"poses_{side}.npy", poses)
        K = np.array([[32.0, 0, 32.0], [0, 32.0, 16.0], [0, 0, 1]], dtype=np.float32)
        np.save(root / f"intrinsics_{side}.npy", np.tile(K, (NUM_FRAMES, 1, 1)))
        for i in range(NUM_FRAMES):
            rgb = rng.integers(0, 255, size=(H, W, 3), dtype=np.uint8)
            Image.fromarray(rgb).save(root / f"rgb_{side}" / f"{i:06d}.png")
            depth = (4.0 + rng.random((H, W))).astype(np.float32)
            np.save(root / f"depth_{side}" / f"{i:06d}.npy", depth)


def _fake_render(gaussians, extrinsics, intrinsics, image_shape, **_):
    b, v = extrinsics.shape[:2]
    h, w = image_shape
    color = torch.sigmoid(gaussians.harmonics[..., 0].mean(1))
    depth = gaussians.means[..., 2].mean(1)
    return {
        "colors": color[:, None, :, None, None].expand(b, v, 3, h, w).contiguous(),
        "depths": depth[:, None, None, None].expand(b, v, h, w),
        "alphas": torch.ones(b, v, h, w),
    }


class _FakeEvaluator:
    def __init__(self, device=None) -> None:
        pass

    def __call__(self, renderer_output, gt):
        mse = (renderer_output["colors"] - gt["original_images"]).square().mean()
        return {"psnr": (-10.0 * torch.log10(mse)).item()}


def _patched_script(record: dict, crash_at_epoch: int | None = None):
    train = _load_train_script()
    train.dataloader_workers = lambda: 0
    train.run_renderer_in_chunk_w_trj_mode = _fake_render
    train.RenderingEvaluator = _FakeEvaluator

    def build_model(cfg, **_):
        backbone = StubBackbone().eval()
        for p in backbone.parameters():
            p.requires_grad_(False)
        return orig_build(cfg, backbone=backbone)

    orig_build = train.build_model
    train.build_model = build_model

    orig_get_loss = train.get_loss

    def get_loss(cfg):
        loss = orig_get_loss(cfg)
        orig_forward = loss.forward

        def forward(*args, **kwargs):
            out = orig_forward(*args, **kwargs)
            record["losses"].append(out["total_loss"].item())
            return out

        loss.forward = forward
        return loss

    train.get_loss = get_loss
    orig_epoch = train.train_epoch

    def train_epoch(model, loader, optimizer, *args, **kwargs):
        epoch = args[4]
        if epoch == crash_at_epoch:
            raise KeyboardInterrupt("simulated interruption")
        record["lrs"].append((epoch, optimizer.param_groups[0]["lr"]))
        return orig_epoch(model, loader, optimizer, *args, **kwargs)

    train.train_epoch = train_epoch
    return train


def _config(data_root: Path, output_dir: Path, epochs: int = 3) -> dict:
    split = data_root / "split.txt"
    split.write_text("scene\n")
    return {
        "seed": 0,
        "max_depth": 80.0,
        "num_views": 2,
        "model": dict(MODEL_CONFIG, cross_view_num_layers=1),
        "training_datasets": {
            "tartanv2": {
                "dataset_path": str(data_root),
                "split_file_path": str(split),
                "img_size": [H, W],
                "stride": 1,
                "num_samples": -1,
            }
        },
        "loss": {
            "name": "l1",
            "color_l1_weight": 0.8,
            "color_ssim_weight": 0.2,
            "color_lpips_weight": 0.0,
            "depth_weight": 0.5,
        },
        "optimization": {
            "batch_size": 2,
            "epochs": epochs,
            "lr": 1e-3,
            "min_lr": 1e-5,
            "warmup_epochs": 1,
            "log_every_n_steps": 1000,
        },
        "logging": {"output_dir": str(output_dir)},
    }


def _final_params(path: Path) -> dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu", weights_only=True)["model_state_dict"]


@pytest.mark.parametrize("grad_accum_steps", [1, 2])
def test_resume_matches_uninterrupted_run(tmp_path, grad_accum_steps) -> None:
    data_root = tmp_path / "data"
    _write_scene(data_root / "scene")

    full = {"losses": [], "lrs": []}
    cfg = _config(data_root, tmp_path / "full")
    cfg["optimization"]["grad_accum_steps"] = grad_accum_steps
    _patched_script(full).train(cfg)

    part = {"losses": [], "lrs": []}
    cfg = _config(data_root, tmp_path / "part")
    cfg["optimization"]["grad_accum_steps"] = grad_accum_steps
    with pytest.raises(KeyboardInterrupt):
        _patched_script(part, crash_at_epoch=1).train(cfg)
    last = tmp_path / "part" / "checkpoints" / "last_checkpoint.pt"
    ckpt = torch.load(last, map_location="cpu", weights_only=True)
    assert ckpt["epoch"] == 0
    steps_per_epoch = len(full["losses"]) // 3
    assert steps_per_epoch == 2  # 5 training windows, batch 2, drop_last
    assert ckpt["global_step"] == steps_per_epoch
    assert ckpt["scheduler_state_dict"]["last_epoch"] == 1
    assert set(ckpt["rng_state"]) == {"torch", "cuda", "numpy", "python"}
    assert (tmp_path / "part" / "checkpoints" / "best_checkpoint.pt").is_file()

    resumed = {"losses": [], "lrs": []}
    cfg = _config(data_root, tmp_path / "part")
    cfg["optimization"]["grad_accum_steps"] = grad_accum_steps
    cfg["resume_from"] = str(last)
    _patched_script(resumed).train(cfg)

    # LR schedule continues where it stopped.
    assert part["lrs"] + resumed["lrs"] == full["lrs"]
    assert [e for e, _ in resumed["lrs"]] == [1, 2]
    # Training losses and final weights match the uninterrupted run exactly.
    assert part["losses"] + resumed["losses"] == full["losses"]
    final_full = tmp_path / "full" / "checkpoints" / "last_checkpoint.pt"
    final_resumed = tmp_path / "part" / "checkpoints" / "last_checkpoint.pt"
    a, b = _final_params(final_full), _final_params(final_resumed)
    assert a.keys() == b.keys()
    for key in a:
        assert torch.equal(a[key], b[key]), key
    ca = torch.load(final_full, map_location="cpu", weights_only=True)
    cb = torch.load(final_resumed, map_location="cpu", weights_only=True)
    assert ca["global_step"] == cb["global_step"] == 3 * steps_per_epoch
    assert ca["epoch"] == cb["epoch"] == 2
    assert ca["best_nvs_psnr"] == cb["best_nvs_psnr"]
    lr_a = ca["optimizer_state_dict"]["param_groups"][0]["lr"]
    assert lr_a == cb["optimizer_state_dict"]["param_groups"][0]["lr"]


def test_resume_rejects_weights_only_checkpoint(tmp_path) -> None:
    train = _load_train_script()
    model = torch.nn.Linear(2, 2)
    path = tmp_path / "weights.pt"
    torch.save({"epoch": 0, "model_state_dict": model.state_dict()}, path)
    opt = torch.optim.AdamW(model.parameters())
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: 1.0)
    with pytest.raises(ValueError, match="init_checkpoint_path"):
        train.resume_training_state(str(path), model, opt, sched)
