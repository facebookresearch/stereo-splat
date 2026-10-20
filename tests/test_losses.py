# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from stereosplat.losses.rendering_losses import (
    SSIM_C1,
    SSIM_C2,
    RenderingLoss,
    gaussian_window,
    ssim,
)


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    a = torch.rand(4, 3, 24, 40, generator=g)
    b = (a + 0.2 * torch.randn(4, 3, 24, 40, generator=g)).clamp(0, 1)
    return a, b


def _reference_ssim(x: torch.Tensor, y: torch.Tensor, k: int = 11) -> torch.Tensor:
    """Direct float64 evaluation of the SSIM map with zero padding."""
    x, y = x.double(), y.double()
    n, c, h, w = x.shape
    g1 = torch.exp(-((torch.arange(k) - k // 2).double() ** 2) / (2 * 1.5**2))
    g1 = g1 / g1.sum()
    weights = torch.outer(g1, g1).reshape(1, 1, k * k, 1)

    def patches(t: torch.Tensor) -> torch.Tensor:
        cols = F.unfold(t.reshape(n * c, 1, h, w), k, padding=k // 2)
        return cols.reshape(n * c, 1, k * k, h * w)

    px, py = patches(x), patches(y)
    mx = (weights * px).sum(2)
    my = (weights * py).sum(2)
    vx = (weights * px * px).sum(2) - mx**2
    vy = (weights * py * py).sum(2) - my**2
    cxy = (weights * px * py).sum(2) - mx * my
    s = ((2 * mx * my + SSIM_C1) * (2 * cxy + SSIM_C2)) / (
        (mx**2 + my**2 + SSIM_C1) * (vx + vy + SSIM_C2)
    )
    return s.reshape(n, c, h, w)


def test_gaussian_window() -> None:
    w = gaussian_window(11, 1.5, 3)
    assert w.shape == (3, 1, 11, 11)
    assert torch.allclose(w.sum(dim=(-1, -2)), torch.ones(3))
    assert torch.equal(w, w.flip(-1)) and torch.equal(w, w.transpose(-1, -2))
    assert w[0, 0].argmax() == 5 * 11 + 5


def test_ssim_matches_direct_formula() -> None:
    a, b = _inputs()
    ref = _reference_ssim(a, b)
    assert torch.allclose(ssim(a, b).double(), ref.mean(), atol=1e-6)
    per_image = ssim(a, b, size_average=False)
    assert per_image.shape == (4,)
    assert torch.allclose(per_image.double(), ref.mean(dim=(1, 2, 3)), atol=1e-6)


def test_ssim_golden_values() -> None:
    # Recorded from the SSIM used to train the released model (fp32, CPU).
    a, b = _inputs()
    assert ssim(a, b).item() == pytest.approx(0.8404346108436584, abs=1e-6)
    expected = [
        0.8350415229797363,
        0.8422349691390991,
        0.8404906392097473,
        0.8439712524414062,
    ]
    assert ssim(a, b, size_average=False).tolist() == pytest.approx(expected, abs=1e-6)
    x = torch.linspace(0, 1, 24 * 40).reshape(1, 1, 24, 40).repeat(2, 3, 1, 1)
    assert ssim(x, 1 - x).item() == pytest.approx(-0.278139591217041, abs=1e-6)
    assert ssim(x, 0.5 * x).item() == pytest.approx(0.6609618663787842, abs=1e-6)


def test_ssim_properties() -> None:
    a, b = _inputs()
    assert ssim(a, a).item() == pytest.approx(1.0, abs=1e-6)
    assert torch.equal(ssim(a, b), ssim(b, a))
    assert ssim(a.double(), b.double()).dtype == torch.float64


def test_rendering_loss_without_lpips() -> None:
    g = torch.Generator().manual_seed(1)
    b, v, h, w = 2, 3, 24, 32
    gt = torch.rand(b, v, 3, h, w, generator=g)
    pred = (gt + 0.1 * torch.randn(b, v, 3, h, w, generator=g)).clamp(0, 1)
    pred.requires_grad_(True)
    depth_gt = 1.0 + torch.rand(b, v, 1, h, w, generator=g)
    depth_pred = depth_gt[:, :, 0] + 0.5
    masks = torch.ones(b, v, 1, h, w)
    masks[0, 0] = 0
    loss_fn = RenderingLoss(
        {
            "color_l1_weight": 0.8,
            "color_ssim_weight": 0.2,
            "color_lpips_weight": 0.0,
            "depth_weight": 0.5,
        }
    )
    out = loss_fn(
        {"colors": pred, "depths": depth_pred},
        {"original_images": gt, "depths": depth_gt, "masks": masks},
    )
    l1 = (pred - gt).abs().mean()
    dssim = 1.0 - ssim(pred.view(-1, 3, h, w), gt.view(-1, 3, h, w))
    assert torch.allclose(out["color_loss"], 0.8 * l1 + 0.2 * dssim)
    assert out["depth_loss"].item() == pytest.approx(0.5, abs=1e-6)
    assert torch.allclose(out["total_loss"], out["color_loss"] + 0.25)
    out["total_loss"].backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
