# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""FoundationStereo adapter: patch checks (fake upstream) and golden values."""

from __future__ import annotations

import os
import types
from pathlib import Path

import pytest
import torch
from torch import nn

from stereosplat.backbones import foundation_stereo as fs_adapter


def _fake_upstream(
    call_normalize: bool = True,
    call_sampler: bool = True,
    call_cnet: bool = True,
) -> tuple[types.ModuleType, types.ModuleType]:
    """Minimal stand-ins for ``core.foundation_stereo`` and ``core.geometry``."""
    geometry = types.ModuleType("fake_geometry")
    geometry.bilinear_sampler = fs_adapter._cudnn_disabled(lambda x: x)
    fs = types.ModuleType("fake_foundation_stereo")
    fs.normalize_image = lambda img: (img - 0.5) / 0.25

    class FoundationStereo(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.args = types.SimpleNamespace(mixed_precision=False, n_gru_layers=3)
            self.feature = nn.Identity()
            self.stem_2 = nn.Conv2d(3, 4, 1)
            self.cnet = _FakeContext()

        def forward(self, image1, image2, iters=12, test_mode=False):
            if call_normalize:
                image1 = fs.normalize_image(image1)
                image2 = fs.normalize_image(image2)
            both = torch.cat([image1, image2], 0)
            vit = self.feature((both, both))
            self.stem_2(image1)
            if call_cnet:
                self.cnet(image1, vit_feat=vit[1][: len(image1)], num_layers=3)
            disp = image1[:, :1] - image2[:, :1]
            if call_sampler:
                disp = geometry.bilinear_sampler(disp)
            return disp

    fs.FoundationStereo = FoundationStereo
    return fs, geometry


class _FakeContext(nn.Module):
    def forward(self, x, vit_feat=None, num_layers=3):
        return [(x, x) for _ in range(num_layers)]


def _run(fs: types.ModuleType, geometry: types.ModuleType, **kwargs):
    model = fs_adapter._make_backbone_class(fs, geometry)().eval()
    x = torch.full((1, 3, 4, 6), 0.75)
    y = torch.full((1, 3, 4, 6), 0.25)
    with torch.no_grad():
        return model(x, y, **kwargs)


def test_patches_take_effect() -> None:
    fs, geometry = _fake_upstream()
    disp, feats = _run(fs, geometry)
    assert torch.allclose(disp, torch.full_like(disp, 0.5))  # inputs not normalized
    assert feats["stem"].shape[0] == 2 and feats["ctx2"].shape[0] == 2
    assert fs.normalize_image(torch.ones(1)).item() == 2.0  # restored afterwards


def test_missing_normalize_call_raises() -> None:
    fs, geometry = _fake_upstream(call_normalize=False)
    with pytest.raises(
        fs_adapter.FoundationStereoAdapterError, match="normalize_image"
    ):
        _run(fs, geometry, return_features=False)


def test_unwrapped_or_unused_sampler_raises() -> None:
    fs, geometry = _fake_upstream(call_sampler=False)
    with pytest.raises(
        fs_adapter.FoundationStereoAdapterError, match="bilinear_sampler"
    ):
        _run(fs, geometry)
    fs, geometry = _fake_upstream()
    geometry.bilinear_sampler = lambda x: x
    with pytest.raises(
        fs_adapter.FoundationStereoAdapterError, match="bilinear_sampler"
    ):
        _run(fs, geometry)


def test_hook_that_did_not_fire_raises() -> None:
    fs, geometry = _fake_upstream(call_cnet=False)
    with pytest.raises(fs_adapter.FoundationStereoAdapterError, match="cnet"):
        _run(fs, geometry)


def test_dinov2_hub_load_is_pinned(monkeypatch) -> None:
    calls = []
    monkeypatch.delenv("STEREOSPLAT_DINOV2_REPO", raising=False)
    monkeypatch.setattr(
        torch.hub, "load", lambda repo, model, *a, **kw: calls.append((repo, model, kw))
    )
    with fs_adapter._offline_construction(None):
        torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True)
    repo, model, kwargs = calls[0]
    assert repo == "facebookresearch/dinov2:7764ea0f912e53c92e82eb78a2a1631e92725fc8"
    assert repo == fs_adapter.DINOV2_HUB_REF
    assert model == "dinov2_vits14"
    assert kwargs["pretrained"] is False and kwargs["skip_validation"] is True


def test_dinov2_not_loaded_through_hub_raises(monkeypatch) -> None:
    monkeypatch.delenv("STEREOSPLAT_DINOV2_REPO", raising=False)
    with pytest.raises(fs_adapter.FoundationStereoAdapterError, match="dinov2"):
        with fs_adapter._offline_construction(None):
            pass


# ---------------------------------------------------------------------------
# Golden values with the real upstream code and the public ViT-S weights.
# Recorded with FoundationStereo from the shriarul5273/FoundationStereo_demo
# mirror (df3ea63) and DINOv2 at DINOV2_COMMIT, CPU fp32, torch 2.4.1.
# ---------------------------------------------------------------------------

GOLDEN = {
    "disp_mean": 7.943528175354004,
    "disp_std": 0.018839344382286072,
    "disp_min": 7.87752628326416,
    "disp_max": 7.9987077713012695,
    "stem_mean": 0.38709378242492676,
    "ctx0_mean": -0.3675406277179718,
    "ctx0_absmean": 2.9724574089050293,
    "ctx1_mean": -0.18537183105945587,
    "ctx1_absmean": 9.357230186462402,
    "ctx2_mean": 0.7509618401527405,
    "ctx2_absmean": 7.052794933319092,
}


def _golden_pair() -> tuple[torch.Tensor, torch.Tensor]:
    """Smooth random texture; the right view is shifted by 8 px (disparity 8)."""
    from stereosplat.utils.preprocessing import get_transforms

    g = torch.Generator().manual_seed(0)
    tex = torch.rand(1, 3, 16, 36, generator=g)
    tex = nn.functional.interpolate(
        tex, size=(64, 144), mode="bilinear", align_corners=False
    )
    normalize = get_transforms().transforms[1]
    return normalize(tex[0, :, :, 0:128])[None], normalize(tex[0, :, :, 8:136])[None]


def _fs_checkpoint_dir() -> Path | None:
    path = os.environ.get("FOUNDATION_STEREO_CKPT_DIR")
    if not path or not (Path(path) / "model_best_bp2.pth").is_file():
        return None
    return Path(path)


@pytest.mark.skipif(
    not os.environ.get("FOUNDATION_STEREO_ROOT") or _fs_checkpoint_dir() is None,
    reason="needs FOUNDATION_STEREO_ROOT and FOUNDATION_STEREO_CKPT_DIR "
    "(cfg.yaml + model_best_bp2.pth of the 11-33-40 release)",
)
def test_foundation_stereo_golden_values() -> None:
    from stereosplat.models.build import build_foundation_stereo_backbone

    backbone = build_foundation_stereo_backbone(str(_fs_checkpoint_dir()))
    left, right = _golden_pair()
    with torch.no_grad():
        disp, feats = backbone(left, right)
    stats = {
        "disp_mean": disp.mean(),
        "disp_std": disp.std(),
        "disp_min": disp.min(),
        "disp_max": disp.max(),
    }
    for key, value in feats.items():
        stats[f"{key}_mean"] = value.mean()
        stats[f"{key}_absmean"] = value.abs().mean()
    for key, expected in GOLDEN.items():
        assert stats[key].item() == pytest.approx(expected, rel=1e-4, abs=1e-5), key
