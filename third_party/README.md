# Third-party code and dependencies

## FoundationStereo (not vendored)

StereoSplat uses the frozen FoundationStereo stereo network
(https://github.com/NVlabs/FoundationStereo, NVIDIA Source Code License,
non-commercial). Its source code is **not** included in or hosted by this
repository; install it yourself from the upstream repository at the commit the
adapter was validated against:

```bash
git clone https://github.com/NVlabs/FoundationStereo third_party/FoundationStereo
git -C third_party/FoundationStereo checkout 6e8806816b533e4d13ddbb95ffa907b797060a62
```

`third_party/FoundationStereo` is the default location (it is listed in
`.gitignore`); a checkout elsewhere can be used with
`export FOUNDATION_STEREO_ROOT=/path/to/FoundationStereo`.
`stereosplat/backbones/foundation_stereo.py` imports the upstream modules and
adapts them at runtime (input normalization, feature taps, offline
construction); see the module docstring. StereoSplat uses the public ViT-S
weights ("11-33-40": `cfg.yaml` + `model_best_bp2.pth`) distributed by NVIDIA.
The DINOv2 backbone code inside FoundationStereo is loaded through `torch.hub`
from https://github.com/facebookresearch/dinov2 (Apache-2.0), pinned to commit
`7764ea0f912e53c92e82eb78a2a1631e92725fc8` (`DINOV2_HUB_REF` in
`stereosplat/backbones/foundation_stereo.py`); set `STEREOSPLAT_DINOV2_REPO` to
use a local checkout of that commit instead. The adapter checks at runtime that
each of its patches took effect and raises `FoundationStereoAdapterError`,
naming the expected upstream symbol, if the FoundationStereo code differs.

## Code adapted from other projects

| Files | Origin | License |
|---|---|---|
| `stereosplat/models/gaussian_adapter.py`, `stereosplat/rendering/renderer.py`, `stereosplat/rendering/gaussians.py`, `stereosplat/utils/geometry.py`, `stereosplat/utils/sh.py` | Depth Anything 3, https://github.com/ByteDance-Seed/Depth-Anything-3 (modified) | Apache-2.0 (header kept in each file) |
| quaternion / rotation helpers in `stereosplat/utils/geometry.py` (via Depth Anything 3) | PyTorch3D, https://github.com/facebookresearch/pytorch3d | BSD-3-Clause |

License texts of the adapted code are in `third_party/licenses/`:
`Apache-2.0.txt` (standard Apache License 2.0 text, as shipped in the gsplat
1.4.0 wheel) and `PyTorch3D-BSD-3-Clause.txt` (from the pytorch3d 0.7.4 wheel
on PyPI).

## Python dependencies

Installed from PyPI (see `requirements.txt`), not redistributed: gsplat
(Apache-2.0), torchmetrics (Apache-2.0; its LPIPS metric ships the LPIPS
weights, BSD-2-Clause), pytorch-msssim (MIT), e3nn (MIT), timm (Apache-2.0),
einops (MIT), OmegaConf (BSD-3-Clause).
