# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os

import pytest
import torch
from conftest import MODEL_CONFIG, StubBackbone, make_window

from stereosplat.models.build import (
    build_model,
    load_checkpoint,
    read_checkpoint,
    remap_state_dict,
    save_release_checkpoint,
)

HEAD_PREFIXES = {
    "block_context0",
    "block_context1",
    "block_context2",
    "block_dec0",
    "block_dec1",
    "block_depth_output",
    "block_raw_gaussians",
    "cross_view_ctx0",
    "cross_view_ctx1",
    "cross_view_ctx2",
    "cross_view_stem",
    "depth_enc1",
    "depth_enc2",
    "depth_enc3",
    "depth_encoder_skip",
    "depth_residual_gate",
    "gaussian_head",
    "image_encoder",
    "log_weight_temperature",
    "weight_head",
}


def test_forward_shapes(stub_model) -> None:
    data = make_window(batch=1, num_pairs=2, h=32, w=64)
    with torch.no_grad():
        out = stub_model(data)
    assert out["depths"].shape == (1, 4, 1, 32, 64)
    assert out["raw_gaussians"].shape == (1, 4, 32, 64, 1 + 3 + 4 + 3)
    assert torch.isfinite(out["depths"]).all()
    assert (out["depths"] > 0).all()
    assert (out["depths"] <= 80.0).all()


def test_forward_gaussians_and_aggregation(stub_model) -> None:
    data = make_window(batch=2, num_pairs=2, h=32, w=64)
    stub_model.merge_distance = 0.05
    with torch.no_grad():
        gaussians = stub_model.forward_gaussians(data)
    num_pixels = 4 * 32 * 64
    assert gaussians.means.shape[0] == 2
    assert 0 < gaussians.means.shape[1] <= num_pixels
    assert gaussians.harmonics.shape[2:] == (3, 1)
    norms = gaussians.rotations.norm(dim=-1)
    valid = gaussians.opacities > 0
    assert torch.allclose(norms[valid], torch.ones_like(norms[valid]), atol=1e-4)


def test_aggregate_requires_forward(stub_model) -> None:
    data = make_window()
    with torch.no_grad():
        out = stub_model(data)
        g = stub_model.gaussian_adapter(
            extrinsics=data["poses"],
            intrinsics=data["intrinsics"],
            depths=out["depths"],
            raw_gaussians=out["raw_gaussians"],
        )
        stub_model.aggregate_gaussians(g)
        with pytest.raises(RuntimeError):
            stub_model.aggregate_gaussians(g)


def test_backward_reaches_head_not_backbone() -> None:
    torch.manual_seed(0)
    backbone = StubBackbone()
    for p in backbone.parameters():
        p.requires_grad_(False)
    model = build_model(MODEL_CONFIG, backbone=backbone).train()
    assert not model.fs_model.training
    data = make_window()
    gaussians = model.forward_gaussians(data)
    loss = gaussians.means.square().mean() + gaussians.opacities.mean()
    loss.backward()
    assert model.depth_residual_gate.grad is not None
    assert model.weight_head.weight.grad is not None
    assert all(p.grad is None for p in model.fs_model.parameters())


def test_state_dict_names() -> None:
    model = build_model(MODEL_CONFIG, backbone=StubBackbone())
    prefixes = {k.split(".")[0] for k in model.state_dict()}
    assert prefixes == HEAD_PREFIXES | {"fs_model"}


def test_checkpoint_roundtrip(tmp_path, stub_model) -> None:
    full = tmp_path / "full.pt"
    torch.save({"epoch": 0, "model_state_dict": stub_model.state_dict()}, full)
    model = build_model(MODEL_CONFIG, backbone=StubBackbone())
    load_checkpoint(model, str(full))
    for k, v in stub_model.state_dict().items():
        assert torch.equal(model.state_dict()[k], v), k

    head = {
        k: v
        for k, v in stub_model.state_dict().items()
        if not k.startswith("fs_model.")
    }
    release = tmp_path / "release.pth"
    torch.save({"version": 1, "config": MODEL_CONFIG, "state_dict": head}, release)
    model = build_model(MODEL_CONFIG, backbone=StubBackbone())
    load_checkpoint(model, str(release))
    assert torch.equal(model.weight_head.weight, stub_model.weight_head.weight)

    bad = dict(head)
    bad.pop("gaussian_head.weight")
    torch.save({"version": 1, "config": MODEL_CONFIG, "state_dict": bad}, release)
    with pytest.raises(RuntimeError):
        load_checkpoint(model, str(release))


def test_safetensors_release_checkpoint(tmp_path, stub_model) -> None:
    head = {
        k: v
        for k, v in stub_model.state_dict().items()
        if not k.startswith("fs_model.")
    }
    config = {k: v for k, v in MODEL_CONFIG.items() if k != "fs_checkpoint_path"}
    path = tmp_path / "release.safetensors"
    save_release_checkpoint(path, head, config)
    ckpt = read_checkpoint(path)
    assert ckpt["version"] == 1
    assert ckpt["config"] == config
    assert ckpt["state_dict"].keys() == head.keys()
    for k, v in head.items():
        assert torch.equal(ckpt["state_dict"][k], v), k
        assert ckpt["state_dict"][k].dtype == v.dtype, k

    model = build_model(MODEL_CONFIG, backbone=StubBackbone())
    meta = load_checkpoint(model, str(path))
    assert meta["version"] == 1
    for k, v in head.items():
        assert torch.equal(model.state_dict()[k], v), k

    bad = dict(head)
    bad.pop("gaussian_head.weight")
    save_release_checkpoint(path, bad, config)
    with pytest.raises(RuntimeError):
        load_checkpoint(model, str(path))

    from safetensors.torch import save_file

    save_file({k: v.contiguous() for k, v in head.items()}, str(path))
    with pytest.raises(RuntimeError, match="metadata"):
        read_checkpoint(path)


def test_remap_strips_ddp_prefix() -> None:
    sd = {"module.weight_head.bias": torch.zeros(1)}
    assert list(remap_state_dict(sd)) == ["weight_head.bias"]


def test_rejects_unsupported_settings() -> None:
    cfg = dict(MODEL_CONFIG, cross_view_mode="concat")
    with pytest.raises(ValueError):
        build_model(cfg, backbone=StubBackbone())
    with pytest.raises(ValueError):
        build_model(dict(MODEL_CONFIG, name="other"), backbone=StubBackbone())


@pytest.mark.skipif(
    not os.environ.get("FOUNDATION_STEREO_ROOT"),
    reason="needs upstream FoundationStereo sources (FOUNDATION_STEREO_ROOT)",
)
def test_foundation_stereo_adapter_contract() -> None:
    from omegaconf import OmegaConf

    from stereosplat.backbones.foundation_stereo import build_foundation_stereo

    cfg_path = os.environ.get("FOUNDATION_STEREO_CFG")
    if cfg_path is None:
        pytest.skip("set FOUNDATION_STEREO_CFG to the checkpoint cfg.yaml")
    backbone = build_foundation_stereo(OmegaConf.load(cfg_path)).eval()
    left = torch.randn(1, 3, 64, 128)
    right = torch.randn(1, 3, 64, 128)
    with torch.no_grad():
        disp, feats = backbone(left, right, iters=2)
    assert disp.shape == (1, 1, 64, 128)
    assert feats["stem"].shape == (2, 32, 32, 64)
    assert feats["ctx0"].shape == (2, 128, 16, 32)
    assert feats["ctx1"].shape == (2, 128, 8, 16)
    assert feats["ctx2"].shape == (2, 128, 4, 8)
    assert all(f.dtype == torch.float32 for f in feats.values())
