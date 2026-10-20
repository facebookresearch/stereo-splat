# Third-party code and dependencies

## FoundationStereo (not vendored)

StereoSplat uses the frozen FoundationStereo stereo network
(https://github.com/NVlabs/FoundationStereo, NVIDIA Source Code License,
non-commercial). Its source code is **not** included in this repository. It is
expected as a git submodule at `third_party/FoundationStereo`:

```bash
git submodule add https://github.com/NVlabs/FoundationStereo third_party/FoundationStereo
git -C third_party/FoundationStereo checkout <pinned commit>
```

or anywhere else via `export FOUNDATION_STEREO_ROOT=/path/to/FoundationStereo`.
`stereosplat/backbones/foundation_stereo.py` imports the upstream modules and
adapts them at runtime (input normalization, feature taps, offline
construction); see the module docstring. StereoSplat uses the public ViT-S
weights ("11-33-40": `cfg.yaml` + `model_best_bp2.pth`) distributed by NVIDIA.
The DINOv2 backbone code inside FoundationStereo is loaded through `torch.hub`
from https://github.com/facebookresearch/dinov2 (Apache-2.0); set
`STEREOSPLAT_DINOV2_REPO` to use a local checkout instead.

## Code adapted from other projects

| Files | Origin | License |
|---|---|---|
| `stereosplat/models/gaussian_adapter.py`, `stereosplat/rendering/renderer.py`, `stereosplat/rendering/gaussians.py`, `stereosplat/utils/geometry.py`, `stereosplat/utils/sh.py` | Depth Anything 3, https://github.com/ByteDance-Seed/Depth-Anything-3 (modified) | Apache-2.0 (header kept in each file) |
| quaternion / rotation helpers in `stereosplat/utils/geometry.py` (via Depth Anything 3) | PyTorch3D, https://github.com/facebookresearch/pytorch3d | BSD-3-Clause |

## Python dependencies

Installed from PyPI (see `requirements.txt`), not redistributed: gsplat
(Apache-2.0), LPIPS (BSD-2-Clause), torchmetrics (Apache-2.0), pytorch-msssim
(MIT), e3nn (MIT), timm (Apache-2.0), einops (MIT), OmegaConf (BSD-3-Clause).
