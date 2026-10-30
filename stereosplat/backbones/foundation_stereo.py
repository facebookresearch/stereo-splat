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
  with the dataset statistics (``stereosplat.utils.preprocessing.get_transforms``),
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
  through ``torch.hub`` from ``facebookresearch/dinov2`` pinned to commit
  ``DINOV2_COMMIT`` (or from a local checkout given by
  ``STEREOSPLAT_DINOV2_REPO``).
* Optional packages imported by the upstream ``Utils.py`` but not needed by
  the network are replaced by placeholders if they are not installed.
* ``bilinear_sampler`` runs ``grid_sample`` with cuDNN disabled, matching the
  setup used to produce the paper results.

Every runtime patch is checked: if the upstream code no longer exposes or
calls the patched symbol (``core.foundation_stereo.normalize_image``,
``core.geometry.bilinear_sampler``, the ``feature`` / ``stem_2`` / ``cnet``
submodules, ``torch.hub.load("facebookresearch/dinov2", ...)``), a
:class:`FoundationStereoAdapterError` naming the symbol is raised instead of
silently computing different features (e.g. normalizing inputs twice).
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
# DINOv2 code revision used to produce the paper results.
DINOV2_COMMIT: str = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
DINOV2_HUB_REF: str = f"{DINOV2_HUB_REPO}:{DINOV2_COMMIT}"

_UPSTREAM: dict[str, types.ModuleType] = {}
_BILINEAR_CALLS: list[int] = [0]


class FoundationStereoAdapterError(RuntimeError):
    """The upstream FoundationStereo code does not match what the adapter patches."""


def _mismatch(symbol: str, detail: str) -> FoundationStereoAdapterError:
    return FoundationStereoAdapterError(
        f"FoundationStereo adapter: {detail} Expected upstream symbol: {symbol}. "
        "The FoundationStereo checkout probably differs from the revision this "
        "adapter was written for (see third_party/README.md)."
    )


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
        _BILINEAR_CALLS[0] += 1
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

    if not callable(getattr(fs_module, "normalize_image", None)):
        raise _mismatch(
            "core.foundation_stereo.normalize_image",
            "input normalization function not found.",
        )
    if not callable(getattr(geometry_module, "bilinear_sampler", None)):
        raise _mismatch(
            "core.geometry.bilinear_sampler", "sampling function not found."
        )
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
    dinov2_loads: list[str] = []

    def create_model(*args: Any, **kwargs: Any) -> Any:
        kwargs["pretrained"] = False
        return orig_create(*args, **kwargs)

    def hub_load(repo_or_dir: str, model: str, *args: Any, **kwargs: Any) -> Any:
        if str(repo_or_dir).split(":")[0] != DINOV2_HUB_REPO:
            return orig_hub_load(repo_or_dir, model, *args, **kwargs)
        dinov2_loads.append(model)
        kwargs["pretrained"] = False
        if local_repo and Path(local_repo).is_dir():
            kwargs["source"] = "local"
            return orig_hub_load(local_repo, model, *args, **kwargs)
        if local_repo:
            # A GitHub "owner/repo[:ref]" given by the user.
            return orig_hub_load(local_repo, model, *args, **kwargs)
        # torch.hub only validates refs that are branch or tag heads; the
        # pinned commit is fixed here, so validation is skipped.
        kwargs["skip_validation"] = True
        return orig_hub_load(DINOV2_HUB_REF, model, *args, **kwargs)

    timm.create_model = create_model
    torch.hub.load = hub_load
    try:
        yield
    finally:
        timm.create_model = orig_create
        torch.hub.load = orig_hub_load
    if not dinov2_loads:
        raise _mismatch(
            f'torch.hub.load("{DINOV2_HUB_REPO}", "dinov2_vit*14", ...) '
            "(depth_anything/dpt.py)",
            "the DINOv2 backbone was not loaded through torch.hub, so the pinned "
            "DINOv2 revision and pretrained=False were not applied.",
        )


@contextlib.contextmanager
def _identity_input_normalization(fs_module: types.ModuleType) -> Iterator[list[int]]:
    """Replace ``normalize_image`` by the identity; yields a call counter."""
    calls = [0]

    def identity(img: torch.Tensor) -> torch.Tensor:
        calls[0] += 1
        return img

    orig = fs_module.normalize_image
    fs_module.normalize_image = identity
    try:
        yield calls
    finally:
        fs_module.normalize_image = orig


def _check_patches_applied(
    geometry_module: types.ModuleType, normalize_calls: int, bilinear_calls: int
) -> None:
    if normalize_calls < 2:
        raise _mismatch(
            "core.foundation_stereo.normalize_image",
            f"the identity replacement was called {normalize_calls} time(s) instead "
            "of once per input image, so FoundationStereo would normalize the "
            "already-normalized inputs again.",
        )
    if not getattr(
        geometry_module.bilinear_sampler, "_stereosplat_cudnn_disabled", False
    ):
        raise _mismatch(
            "core.geometry.bilinear_sampler",
            "the cuDNN-disabled wrapper was replaced.",
        )
    if bilinear_calls == 0:
        raise _mismatch(
            "core.geometry.bilinear_sampler",
            "the cuDNN-disabled wrapper was never called during the forward pass.",
        )


def _make_backbone_class(
    fs_module: types.ModuleType, geometry_module: types.ModuleType
) -> type[nn.Module]:
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
                for attr, name in (
                    ("feature", "feature"),
                    ("stem_2", "stem"),
                    ("cnet", "cnet"),
                ):
                    module = getattr(self, attr, None)
                    if not isinstance(module, nn.Module):
                        raise _mismatch(
                            f"FoundationStereo.{attr}", "submodule not found."
                        )
                    handles.append(module.register_forward_hook(_capture(name)))
            bilinear_before = _BILINEAR_CALLS[0]
            try:
                with _identity_input_normalization(fs_module) as normalize_calls:
                    disp = super().forward(image1, image2, iters=iters, test_mode=True)
            finally:
                for h in handles:
                    h.remove()
            _check_patches_applied(
                geometry_module,
                normalize_calls[0],
                _BILINEAR_CALLS[0] - bilinear_before,
            )
            if not return_features:
                return disp, None
            for name, attr in (
                ("feature", "feature"),
                ("stem", "stem_2"),
                ("cnet", "cnet"),
            ):
                if name not in captured:
                    raise _mismatch(
                        f"FoundationStereo.{attr}",
                        "the feature hook did not fire during the forward pass.",
                    )

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
        dinov2_repo: Optional local DINOv2 checkout (or torch.hub
            ``owner/repo:ref``) used instead of ``DINOV2_HUB_REF``.
    """
    modules = import_upstream(root)
    backbone_cls = _make_backbone_class(
        modules["foundation_stereo"], modules["geometry"]
    )
    with _offline_construction(dinov2_repo):
        return backbone_cls(cfg)
