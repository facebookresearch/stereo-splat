<p align="center">

  <h1 align="center">StereoSplat: Metric-Scale Novel View Synthesis via Stereo-Grounded Gaussian Splatting</h1>
  <!-- TODO(owner): author page links, the first name of Vansteenkiste, and the affiliations line. -->
  <p align="center">
    <strong>Vladimir Yugay</strong>
    &middot;
    <strong>Denis Rozumny</strong>
    &middot;
    <strong>Theo Gevers</strong>
    &middot;
    <strong>Vansteenkiste</strong>
    &middot;
    <strong>Martin R. Oswald</strong>
    &middot;
    <strong>Diogo Luvizon</strong>
  </p>
  <p align="center">NeurIPS 2026</p>
  <!-- TODO(owner): paper, project page and Hugging Face links (placeholders below). -->
  <h3 align="center"><a href="#">Paper</a> | <a href="#">Project Page</a> | <a href="https://huggingface.co/facebook/stereo-splat">Model Weights</a></h3>
  <div align="center"></div>
</p>

<p align="center">
  <img src="./assets/teaser.png" alt="StereoSplat teaser: input left image, two rendered novel views and their depth maps" width="90%">
</p>
<p align="center"><sub>Single stereo pair from <a href="https://tartanair.org">TartanAir V2</a> (AbandonedFactory2) -> metric 3D Gaussians;
novel views rendered 0.5 m to the side and 0.5 m forward. Produced with <code>scripts/demo.py</code>.</sub></p>

StereoSplat reconstructs metric-scale 3D Gaussians from a short sequence of
calibrated stereo pairs in a single feed-forward pass, and renders novel views
and depth. Geometry comes from a frozen stereo backbone,
[FoundationStereo](https://github.com/NVlabs/FoundationStereo). A light decoder
with cross-view attention predicts per-pixel Gaussians, which are aggregated
across views with learned, depth-adaptive weights. Stereo gives metric depth, so
the reconstructions and the rendered depth are in meters, and the model works
with poses from a stereo visual odometry system such as
[MAC-VO](https://github.com/MAC-VO/MAC-VO) instead of ground truth.

## &#x1F4CA; Results

Novel view synthesis with the released weights and MAC-VO poses, following the
paper protocol: windows of 3 stereo pairs, and the middle pair is rendered. We
produced these numbers with this repository. They match Table 1 of the paper
after rounding.

| Benchmark | PSNR | SSIM | LPIPS | AbsRel | delta1 (%) |
|---|---|---|---|---|---|
| Infinigen | 29.41 | 0.954 | 0.116 | 0.033 | 98.56 |
| TartanAir v2 | 26.25 | 0.944 | 0.101 | 0.058 | 94.95 |
| KITTI (zero-shot) | 25.20 | 0.941 | 0.101 | 0.044 | 95.98 |

SSIM is multi-scale SSIM (`sssim` in `metrics.json`). AbsRel and delta1 are
computed on the rendered depth of the novel view.

Depth accuracy (delta1, %) against feed-forward baselines, which use
ground-truth poses:

| Benchmark | StereoSplat | DepthSplat | pixelSplat |
|---|---|---|---|
| Infinigen | 98.6 | 63.0 | 40.1 |
| TartanAir v2 | 94.9 | 78.7 | - |
| KITTI (zero-shot) | 96.0 | 79.8 | - |

## &#x2699;&#xFE0F; Setting Things Up

Clone the repo:

```
git clone https://github.com/facebookresearch/stereo-splat.git
cd stereo-splat
```

Clone FoundationStereo at the commit we tested. NVIDIA releases it under its own
license, so it is not part of this repo:

```
git clone https://github.com/NVlabs/FoundationStereo third_party/FoundationStereo
git -C third_party/FoundationStereo checkout 6e8806816b533e4d13ddbb95ffa907b797060a62
```

`third_party/FoundationStereo` is the default location. To use a checkout
elsewhere, set `FOUNDATION_STEREO_ROOT=/path/to/FoundationStereo`. See
[`third_party/README.md`](third_party/README.md) for details.

Run the following commands to set up the environment:

```
conda create -n stereosplat python=3.10 -y
conda activate stereosplat
```

Install PyTorch. We tested with the CUDA 12.1 build:

```
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
```

Install the other dependencies and the package:

```
pip install -r requirements.txt
pip install -e .
```

A regular `pip install .` also works, since the split files in `data/splits`
are installed with the package. In that case set `FOUNDATION_STEREO_ROOT`,
because `third_party/` is not part of the installed package.

`gsplat` 1.4.0 from PyPI compiles its CUDA kernels the first time it is used.
This takes a few minutes, and the result is cached in
`~/.cache/torch_extensions`. It needs a CUDA 12 toolkit: `nvcc` on `PATH` and
`CUDA_HOME` pointing to a toolkit that includes the CUDA runtime headers:

```
export CUDA_HOME=/usr/local/cuda-12.1 PATH=/usr/local/cuda-12.1/bin:$PATH
python -c "import torch, gsplat; print(gsplat.__version__)"
```

We tested the build with `nvcc` 12.8 against the CUDA 12.1 PyTorch wheels. Set
`TORCH_CUDA_ARCH_LIST` (e.g. `8.0` for A100) to compile only for your GPU. You
can also install a prebuilt `gsplat` 1.4.0 wheel that matches your PyTorch and
CUDA versions (see the [gsplat documentation](https://docs.gsplat.studio/)).

The DINOv2 code inside FoundationStereo is fetched with `torch.hub` from
[facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) at commit
`7764ea0f912e53c92e82eb78a2a1631e92725fc8` the first time the model is built.
To work offline, set `STEREOSPLAT_DINOV2_REPO` to a local dinov2 checkout of that
commit. No pretrained weights are downloaded through `torch.hub` or `timm`; all
backbone weights come from the FoundationStereo checkpoint.

All configs read paths from two environment variables:

```
export DATA_ROOT=/path/to/processed_datasets   # see "Getting the Data"
export CKPT_ROOT=$PWD/checkpoints              # see "Checkpoints"
```

Run the unit tests (CPU) with `pip install pytest && pytest tests`.

We tested our code on NVIDIA A100 GPUs with Linux, Python 3.10, PyTorch 2.4.1,
torchvision 0.19.1 and gsplat 1.4.0 (pinned versions in
[`requirements.txt`](requirements.txt)). Evaluation and rendering in the demo
need a CUDA GPU.

## &#x1F528; Running StereoSplat

<details>
<summary><b>Getting the Data</b></summary>

Every dataset is converted to the same layout: one folder per sequence with
per-frame PNG images and `.npy` depth maps, plus camera-to-world poses and
intrinsics. [`scripts/prepare_data/README.md`](scripts/prepare_data/README.md)
has the download links, the expected raw layout and the conversion commands.

| Dataset | Role | Folder under `${DATA_ROOT}` | Split file |
|---|---|---|---|
| Infinigen stereo video (InfinigenSV) | evaluation | `infinigen_processed/` | `data/splits/infinigen_static/test.txt` |
| TartanAir v2 | evaluation | `tartanair_v2_processed/` | `data/splits/tartanv2/test.txt` |
| KITTI Visual Odometry (static segments) | zero-shot evaluation | `kitti_static/` | `data/splits/kitti_static/test.txt` |
| MAC-VO poses (optional) | external poses for evaluation | `macvo_poses/` | see [`docs/macvo_poses.md`](docs/macvo_poses.md) |

The Infinigen test scenes are stereo video sequences of the public Hugging Face
dataset [`MatchLab/Infinigen`](https://huggingface.co/datasets/MatchLab/Infinigen)
(InfinigenSV, CC BY-NC-SA 4.0).
TODO(owner): confirm that the paper's Infinigen numbers are reproducible from
these public scenes.

Relative split-file paths in the configs are resolved against the current
directory first and then the repository root. Evaluation stops with an error if
a scene listed in a split file is missing under `DATA_ROOT` or a dataset ends up
empty, and logs the number of scenes and windows per dataset at startup. To run
on a deliberately partial copy of the data, set `allow_missing_scenes: true`
(globally or per dataset) or pass `--allow-missing-scenes`.
</details>

<details>
<summary><b>Checkpoints</b></summary>

The released checkpoint holds the StereoSplat decoder, plus the few
FoundationStereo BatchNorm statistics that differ from the public
FoundationStereo weights. All other FoundationStereo weights are loaded from
NVIDIA's release. The training code is not part of this release.

```
bash scripts/download_weights.sh        # into ${CKPT_ROOT:-checkpoints}
```

The script downloads and verifies `stereosplat_v1.pth` from Hugging Face, and
prints instructions for the FoundationStereo weights. The same weights are also
provided as `stereosplat_v1.safetensors`, with the model config in the file
metadata. `--checkpoint` accepts either file.
TODO(owner): Hugging Face repo id, file name and sha256 in
`scripts/download_weights.sh`.

Download the FoundationStereo **ViT-S model `11-33-40`** from the links in the
[FoundationStereo README](https://github.com/NVlabs/FoundationStereo). NVIDIA
releases it under its own license, and it is not redistributed here. Expected
layout:

```
${CKPT_ROOT}/
  stereosplat_v1.pth
  foundation_stereo/11-33-40/
    cfg.yaml
    model_best_bp2.pth   # sha256 8d7850b9dc68d1366722a02a39745704b1db41471211be6abd93ef463727e6be
```

The configs point to this folder with
`model.fs_checkpoint_path: ${oc.env:CKPT_ROOT}/foundation_stereo/11-33-40`.
</details>

<details>
<summary><b>Evaluation</b></summary>

Evaluation uses windows of 3 consecutive stereo pairs. The first and last pairs
are the input, and the middle pair is rendered and compared with the ground
truth: PSNR, SSIM, LPIPS, and AbsRel / delta1 of the rendered depth.

```
python scripts/evaluate.py --config configs/eval_tartanv2.yaml \
    --checkpoint ${CKPT_ROOT}/stereosplat_v1.pth \
    --poses-dir ${DATA_ROOT}/macvo_poses \
    --output-dir outputs/eval_tartanv2
```

Use `configs/eval_kitti.yaml` and `configs/eval_infinigen.yaml` for the other
benchmarks. `torchrun --nproc_per_node=N scripts/evaluate.py ...` splits the
test set across GPUs.

With `--poses-dir`, every dataset is evaluated twice: with ground-truth poses
(`<dataset>_NVS_gt`) and with the external poses
`<poses_dir>/<dataset>/<sequence>/<pose_type>.npy` (`<dataset>_NVS_ext`, the
paper protocol for StereoSplat). Without `--poses-dir`, only ground-truth poses
are used.

Other options: `--pose-type` (default `poses_left_aligned_scaled`, used for all
paper numbers), `--pose-modes gt,ext`, `--num-views`, `--batch-size`,
`--scale-align-depth` (median-scale alignment before the depth metrics; off for
the paper numbers), `--profile-runtime` and OmegaConf dot-list overrides. A
summary is printed at the end and, with `--output-dir`, written to
`metrics.json`, plus TensorBoard logs in `tensorboard/`.

[`docs/macvo_poses.md`](docs/macvo_poses.md) describes how to produce the MAC-VO
poses, and their exact format and alignment.
TODO(owner): link the precomputed MAC-VO poses for the test splits if their
release is approved.
</details>

<details>
<summary><b>Demo</b></summary>

Reconstruct Gaussians from a rectified stereo pair and render novel views. This
needs a CUDA GPU; add `--no-render` to only export the `.ply`, which also works
on CPU:

```
python scripts/demo.py \
    --left path/to/left.png --right path/to/right.png \
    --intrinsics FX FY CX CY --baseline 0.25 \
    --checkpoint ${CKPT_ROOT}/stereosplat_v1.pth \
    --fs-checkpoint-dir ${CKPT_ROOT}/foundation_stereo/11-33-40 \
    --output-dir outputs/demo
```

For a short stereo sequence, put the images in `<dir>/left/` and `<dir>/right/`
(paired in sorted order), and pass metric left-camera camera-to-world poses
(OpenCV axes; `.npy` of shape `(N, 4, 4)` or a text file with 12 or 16 values
per line):

```
python scripts/demo.py --sequence-dir path/to/seq --poses path/to/seq/poses_left.npy \
    --max-frames 4 --stride 1 \
    --intrinsics FX FY CX CY --baseline 0.25 \
    --checkpoint ${CKPT_ROOT}/stereosplat_v1.pth --output-dir outputs/demo_seq
```

Intrinsics are in pixels of the input images. Images are resized and
center-cropped to `--img-size` (default `320 640`, the training resolution).

The demo writes:

<ul>
<li><code>gaussians.ply</code> in the standard 3D Gaussian Splatting layout, which
opens in common 3DGS viewers (<code>--ply-sh-degree</code>, default 3, zero-pads the
higher SH orders);</li>
<li>the predicted depth of the input views;</li>
<li>rendered RGB and depth for the first input camera and for
<code>--num-views</code> cameras on a small orbit (<code>--orbit-radius</code>,
default the baseline; <code>--forward</code>);</li>
<li><code>cameras.json</code>.</li>
</ul>

`--checkpoint` and `--fs-checkpoint-dir` default to the `CKPT_ROOT` layout from
"Checkpoints".
</details>

## &#x1F4CC; Citation

If you find our paper and code useful, please cite us:

<!-- TODO(owner): verify author first names and add the arXiv/proceedings reference. -->

```bib
@inproceedings{yugay2026stereosplat,
  title     = {StereoSplat: Metric-Scale Novel View Synthesis via Stereo-Grounded Gaussian Splatting},
  author    = {Yugay, Vladimir and Rozumny, Denis and Gevers, Theo and Vansteenkiste, TODO(owner) and Oswald, Martin R. and Luvizon, Diogo},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## &#x1F4DC; License

StereoSplat code and model weights are released under the CC BY-NC 4.0
license, see [LICENSE](LICENSE).

Third-party components keep their own licenses:

- FoundationStereo code and weights (NVIDIA, non-commercial license) are not
  redistributed. You get them from the upstream repository and NVIDIA's
  release.
- Files adapted from Depth Anything 3 (Apache-2.0) keep their original
  copyright headers and are marked as modified (see
  [`third_party/README.md`](third_party/README.md) and [NOTICE](NOTICE)).
- Datasets are not redistributed and are subject to their own terms (see
  [`scripts/prepare_data/README.md`](scripts/prepare_data/README.md#licenses)).

## &#x1F64F; Acknowledgements

This work builds on several open-source projects, and we thank their authors:

- [FoundationStereo](https://github.com/NVlabs/FoundationStereo) for the stereo
  backbone and pretrained weights.
- [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3), from
  which parts of the Gaussian adapter, renderer, spherical-harmonics and
  geometry utilities are adapted.
- [gsplat](https://github.com/nerfstudio-project/gsplat) for differentiable
  Gaussian rasterization.
- [MAC-VO](https://github.com/MAC-VO/MAC-VO) for the stereo visual odometry
  poses used in evaluation.

We also thank the creators of TartanAir, TartanAir v2, KITTI, Infinigen and
InfinigenSV for making their data available.
