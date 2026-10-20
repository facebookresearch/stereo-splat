# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Adapter around the upstream NVlabs FoundationStereo implementation.

FoundationStereo (https://github.com/NVlabs/FoundationStereo) is distributed
under the NVIDIA Source Code License and is NOT vendored in this repository.
Clone it into ``third_party/FoundationStereo`` (git submodule) or point the
``FOUNDATION_STEREO_ROOT`` environment variable at a checkout. This module only
imports the upstream classes and wraps them; no upstream code is copied.

StereoSplat needs a few behaviours that differ from the upstream defaults.
They are all applied from the outside:

* Input normalization: StereoSplat feeds images that are already normalized
  with the dataset statistics (``stereosplat.utils.training.get_transforms``),
  so the upstream ``normalize_image`` call inside ``forward`` is replaced by the
  identity for the duration of our forward pass.
* Feature taps: StereoSplat consumes the 1/2-resolution stem features and the
  context-network features of BOTH stereo views. The left-view tensors are
  captured with forward hooks during the upstream forward; the right-view
  tensors are computed afterwards by running the same ``stem_2`` / ``cnet``
  submodules on the right image (with the right half of the ViT features
  captured from ``feature``). Features are returned in float32.
* Offline construction: the backbone weights come from the FoundationStereo
  checkpoint, so the ImageNet EdgeNeXt weights (``timm``) and the DINOv2 hub
  weights are never downloaded (``pretrained=False``). DINOv2 code is loaded
  through ``torch.hub`` from ``facebookresearch/dinov2`` (or from a local
  checkout given by ``STEREOSPLAT_DINOV2_REPO``).
* Optional packages imported by the upstream ``Utils.py`` but not needed by
  the network are replaced by placeholders if they are not installed.
* ``bilinear_sampler`` runs ``grid_sample`` with cuDNN disabled, matching the
  setup used to produce the paper results.
"""

from __future__ import annotations

import contextlib
import functools
import os
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from torch import nn

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_FOUNDATION_STEREO_ROOT: Path = REPO_ROOT / "third_party" / "FoundationStereo"
DINOV2_HUB_REPO: str = "facebookresearch/dinov2"

_UPSTREAM: dict[str, types.ModuleType] = {}


def _resolve_root(root: str | os.PathLike[str] | None) -> Path:
    candidate = root or os.environ.get("FOUNDATION_STEREO_ROOT")
    path = Path(candidate) if candidate else DEFAULT_FOUNDATION_STEREO_ROOT
    if not (path / "core" / "foundation_stereo.py").is_file():
        raise ImportError(
            f"FoundationStereo sources not found at {path}. Run "
            "`git submodule update --init third_party/FoundationStereo` or set "
            "FOUNDATION_STEREO_ROOT to a clone of "
            "https://github.com/NVlabs/FoundationStereo."
        )
    return path.resolve()


def _ensure_utils_module(max_stubs: int = 32) -> list[str]:
    """Import the upstream top-level ``Utils`` module.

    ``core`` star-imports ``Utils``, which also pulls in visualization / IO
    packages that the network never uses (e.g. open3d, trimesh). Missing
    packages are replaced by empty placeholder modules while ``Utils`` is
    imported and removed from ``sys.modules`` afterwards.

    Returns:
        Names of the packages that were replaced by placeholders.
    """
    stubbed: list[str] = []
    try:
        for _ in range(max_stubs):
            try:
                import Utils  # noqa: F401

                return stubbed
            except ModuleNotFoundError as e:
                if not e.name or e.name in stubbed or e.name == "Utils":
                    raise
                sys.modules[e.name] = types.ModuleType(e.name)
                stubbed.append(e.name)
                sys.modules.pop("Utils", None)
        raise ImportError(
            f"Could not import FoundationStereo Utils (stubbed {stubbed})"
        )
    finally:
        for name in stubbed:
            sys.modules.pop(name, None)


def _cudnn_disabled(fn: Any) -> Any:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with torch.backends.cudnn.flags(enabled=False):
            return fn(*args, **kwargs)

    wrapper._stereosplat_cudnn_disabled = True  # type: ignore[attr-defined]
    return wrapper


def import_upstream(
    root: str | os.PathLike[str] | None = None,
) -> dict[str, types.ModuleType]:
    """Import the upstream FoundationStereo modules (cached)."""
    if _UPSTREAM:
        return _UPSTREAM
    path = str(_resolve_root(root))
    if path not in sys.path:
        sys.path.insert(0, path)
    _ensure_utils_module()
    import core.foundation_stereo as fs_module
    import core.geometry as geometry_module

    if not getattr(
        geometry_module.bilinear_sampler, "_stereosplat_cudnn_disabled", False
    ):
        geometry_module.bilinear_sampler = _cudnn_disabled(
            geometry_module.bilinear_sampler
        )
    _UPSTREAM["foundation_stereo"] = fs_module
    _UPSTREAM["geometry"] = geometry_module
    return _UPSTREAM


@contextlib.contextmanager
def _offline_construction(dinov2_repo: str | None) -> Iterator[None]:
    """Build the network without downloading pretrained ImageNet/DINOv2 weights."""
    import timm

    orig_create = timm.create_model
    orig_hub_load = torch.hub.load
    local_repo = dinov2_repo or os.environ.get("STEREOSPLAT_DINOV2_REPO")

    def create_model(*args: Any, **kwargs: Any) -> Any:
        kwargs["pretrained"] = False
        return orig_create(*args, **kwargs)

    def hub_load(repo_or_dir: str, model: str, *args: Any, **kwargs: Any) -> Any:
        if repo_or_dir == DINOV2_HUB_REPO:
            kwargs["pretrained"] = False
            if local_repo and Path(local_repo).is_dir():
                kwargs["source"] = "local"
                return orig_hub_load(local_repo, model, *args, **kwargs)
            return orig_hub_load(local_repo or repo_or_dir, model, *args, **kwargs)
        return orig_hub_load(repo_or_dir, model, *args, **kwargs)

    timm.create_model = create_model
    torch.hub.load = hub_load
    try:
        yield
    finally:
        timm.create_model = orig_create
        torch.hub.load = orig_hub_load


@contextlib.contextmanager
def _identity_input_normalization(fs_module: types.ModuleType) -> Iterator[None]:
    orig = fs_module.normalize_image
    fs_module.normalize_image = lambda img: img
    try:
        yield
    finally:
        fs_module.normalize_image = orig


def _make_backbone_class(fs_module: types.ModuleType) -> type[nn.Module]:
    base = fs_module.FoundationStereo

    class FoundationStereoBackbone(base):  # type: ignore[misc, valid-type]
        """Upstream FoundationStereo returning ``(disparity, features)``.

        ``features`` holds ``stem``, ``ctx0``, ``ctx1``, ``ctx2`` with the left
        and right views concatenated along the batch dimension (float32).
        Inputs must already be normalized (see module docstring).
        """

        def forward(  # type: ignore[override]
            self,
            image1: torch.Tensor,
            image2: torch.Tensor,
            iters: int = 12,
            return_features: bool = True,
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
            captured: dict[str, Any] = {}

            def _capture(name: str) -> Any:
                def hook(_m: nn.Module, _i: Any, out: Any) -> None:
                    captured.setdefault(name, out)

                return hook

            handles = []
            if return_features:
                handles = [
                    self.feature.register_forward_hook(_capture("feature")),
                    self.stem_2.register_forward_hook(_capture("stem")),
                    self.cnet.register_forward_hook(_capture("cnet")),
                ]
            try:
                with _identity_input_normalization(fs_module):
                    disp = super().forward(image1, image2, iters=iters, test_mode=True)
            finally:
                for h in handles:
                    h.remove()
            if not return_features:
                return disp, None

            batch = len(image1)
            vit_feat = captured["feature"][1]
            with torch.amp.autocast("cuda", enabled=self.args.mixed_precision):
                stem_right = self.stem_2(image2)
                cnet_right = self.cnet(
                    image2,
                    vit_feat=vit_feat[batch:],
                    num_layers=self.args.n_gru_layers,
                )
            cnet_left = captured["cnet"]
            features = {
                "stem": torch.cat([captured["stem"], stem_right], dim=0).float(),
                "ctx0": torch.cat([cnet_left[0][0], cnet_right[0][0]], dim=0).float(),
                "ctx1": torch.cat([cnet_left[1][0], cnet_right[1][0]], dim=0).float(),
                "ctx2": torch.cat([cnet_left[2][0], cnet_right[2][0]], dim=0).float(),
            }
            return disp, features

    return FoundationStereoBackbone


def build_foundation_stereo(
    cfg: Any,
    root: str | os.PathLike[str] | None = None,
    dinov2_repo: str | None = None,
) -> nn.Module:
    """Construct the FoundationStereo backbone (random init, no downloads of weights).

    Args:
        cfg: The ``cfg.yaml`` shipped with the FoundationStereo checkpoint,
            loaded with ``OmegaConf.load``.
        root: FoundationStereo checkout (default: ``FOUNDATION_STEREO_ROOT`` or
            ``third_party/FoundationStereo``).
        dinov2_repo: Optional local DINOv2 checkout used instead of the
            ``facebookresearch/dinov2`` torch.hub repository.
    """
    modules = import_upstream(root)
    backbone_cls = _make_backbone_class(modules["foundation_stereo"])
    with _offline_construction(dinov2_repo):
        return backbone_cls(cfg)
