# Data preparation

StereoSplat reads every dataset from a common "processed" layout: one folder
per sequence with per-frame PNG images and `.npy` depth maps, plus one `.npy`
file each for camera-to-world poses and intrinsics. The scripts in this folder
convert the official public downloads into that layout. They only read from
`--input-dir` and write to `--output-dir`; no network access is needed except
for downloading the raw data yourself.

| Dataset | Used for | Script | Processed root expected by the configs |
|---|---|---|---|
| TartanAir (v1) | training | `tartan.py` | `${DATA_ROOT}/tartanair_processed` |
| Infinigen stereo video (InfinigenSV) | training + evaluation | `infinigen.py` | `${DATA_ROOT}/infinigen_processed` |
| TartanAir v2 | evaluation | `tartanv2.py` | `${DATA_ROOT}/tartanair_v2_processed` |
| KITTI Visual Odometry | evaluation (zero-shot) | `kitti.py` | `${DATA_ROOT}/kitti_static` |
| XRStereo | training | none (extract the release) | `${DATA_ROOT}/xrstereo/sequences` |

These folder names are the `dataset_path` values used by
`configs/train_stereosplat.yaml` and `configs/eval_*.yaml`; if you use other
locations, override them on the command line, e.g.
`test_datasets.kitti.dataset_path=/other/kitti_static`.

Requirements: the main package (`pip install -r requirements.txt`; the scripts
use `numpy`, `pillow`, `scipy>=1.14`, `opencv-python-headless` and `tqdm`). For
downloading from Hugging Face: the `hf` command of `huggingface_hub>=1.0`
(installed with the requirements). Pass one `--include` per pattern;
`--include a b` treats `b` as a file name.

All conversions are deterministic and resumable: a folder that already exists in
the output is not rewritten (delete it to regenerate). Each script prints a
warning and exits with a non-zero code if a sequence is incomplete (e.g. frame
counts of images, depth maps and poses differ). Every script accepts
`--split-file data/splits/<name>/<split>.txt` to convert only the sequences
used by a split.

---

## TartanAir v1 (training)

Source: https://theairlab.org/tartanair-dataset/ , mirrored on Hugging Face as
[`theairlabcmu/tartanair`](https://huggingface.co/datasets/theairlabcmu/tartanair).
Training uses all 369 trajectories of the 18 environments (`Easy` and `Hard`;
`data/splits/tartan/train.txt`). Only the left/right image and depth archives
are needed: about 419 GB for all environments (flow and segmentation archives
are not used). The zips are read directly, without extracting them.

The converter keeps every 5th frame by default (`--keep-every 5`), the frames
that training reads (`stride: 5` in the config), so the processed data takes
about 189 GB instead of about 940 GB. To limit the space needed for the raw
archives, download and convert one environment at a time and delete its zips
afterwards (about 59 GB for the largest environment):

```bash
ENVS=$(sed -E 's/_(Easy|Hard)_P[0-9]+$//' data/splits/tartan/train.txt | sort -u)
for env in $ENVS; do
  inc=()
  for d in Easy Hard; do
    for z in image_left image_right depth_left depth_right; do
      inc+=(--include "$env/$d/$z.zip")
    done
  done
  hf download theairlabcmu/tartanair --repo-type dataset \
      --local-dir /data/raw/tartanair "${inc[@]}"
  python scripts/prepare_data/tartan.py \
      --input-dir /data/raw/tartanair --scenes "$env" \
      --output-dir ${DATA_ROOT}/tartanair_processed \
      --split-file data/splits/tartan/train.txt --workers 8 \
    && rm -rf "/data/raw/tartanair/$env"
done
```

Expected raw tree:

```
/data/raw/tartanair/
  abandonedfactory/
    Easy/{image_left,image_right,depth_left,depth_right}.zip
    Hard/{image_left,image_right,depth_left,depth_right}.zip
  abandonedfactory_night/
  ...
```

Result:

```
tartanair_processed/
  abandonedfactory_Easy_P000/
    rgb_left/000000.png ...          640x480 RGB (frames 0, 5, 10, ... of the trajectory)
    rgb_right/000000.png ...
    depth_left/000000.npy ...        (480, 640) float32, meters
    depth_right/000000.npy ...
    poses_left.npy                   (N, 4, 4) float32 camera-to-world, OpenCV axes
    poses_right.npy
    intrinsics_left.npy              (3, 3) float32 (fx = fy = 320, cx = 320, cy = 240)
    intrinsics_right.npy
    keep_every.txt                   frame subsampling factor (5)
```

`--keep-every 1` keeps all frames; the loader reads `keep_every.txt` and
samples the same frames for a given `stride` either way.

Poses: TartanAir stores `tx ty tz qx qy qz qw` in a NED world frame with an
x-forward camera; the script converts both frames to the OpenCV convention
(x right, y down, z forward), so the right camera sits at +0.25 m along the left
camera's x axis.

## TartanAir v2 (evaluation)

Source: https://tartanair.org , mirrored on Hugging Face as
[`theairlabcmu/tartanair2`](https://huggingface.co/datasets/theairlabcmu/tartanair2).
Only the front pinhole stereo pair (`lcam_front`, `rcam_front`) is used. The
test split (`data/splits/tartanv2/test.txt`, 14 trajectories) needs about 52 GB
of archives:

```bash
SPLIT=data/splits/tartanv2/test.txt
INCLUDES=$(awk -F_ '{d=$(NF-1); env=$0; sub("_"d"_"$NF"$","",env);
                    print env"/Data_"d}' "$SPLIT" | sort -u |
           while read p; do
             for z in image_lcam_front image_rcam_front depth_lcam_front depth_rcam_front; do
               echo "$p/$z.zip"
             done
           done)
inc=()
for f in $INCLUDES; do inc+=(--include "$f"); done
hf download theairlabcmu/tartanair2 --repo-type dataset \
    --local-dir /data/raw/tartanair_v2 "${inc[@]}"
```

Expected raw tree:

```
/data/raw/tartanair_v2/
  AbandonedFactory/
    Data_easy/{image,depth}_{lcam,rcam}_front.zip
    Data_hard/...
  ...
```

Convert:

```bash
python scripts/prepare_data/tartanv2.py \
    --input-dir /data/raw/tartanair_v2 \
    --output-dir ${DATA_ROOT}/tartanair_v2_processed \
    --split-file data/splits/tartanv2/test.txt
```

Result:

```
tartanair_v2_processed/
  AbandonedFactory2_hard_P004/
    rgb_lcam_front/000000.png ...    640x640 RGB (PNG bytes copied unchanged)
    rgb_rcam_front/000000.png ...
    depth_lcam_front/000000.npy ...  (640, 640) float32, meters (decoded from RGBA PNG)
    depth_rcam_front/000000.npy ...
    poses_lcam_front.npy             (N, 4, 4) float32 camera-to-world, OpenCV axes
    poses_rcam_front.npy
    intrinsics_lcam_front.npy        (3, 3) float32 (fx = fy = cx = cy = 320)
    intrinsics_rcam_front.npy
```

## KITTI Visual Odometry (evaluation, zero-shot)

Source: the KITTI Visual Odometry / SLAM benchmark,
https://www.cvlibs.net/datasets/kitti/eval_odometry.php (free registration
required). Download these four archives into one folder:

| File | Content | Size (approx.) |
|---|---|---|
| `data_odometry_color.zip` | color stereo images `image_2` (left), `image_3` (right), `times.txt` | 65 GB |
| `data_odometry_velodyne.zip` | LiDAR scans (sparse depth for the depth metrics) | 80 GB |
| `data_odometry_calib.zip` | `calib.txt` (P0-P3, Tr) | < 1 MB |
| `data_odometry_poses.zip` | ground-truth poses for sequences 00-10 | < 5 MB |

Expected raw tree (after `--extract`, or extract the zips yourself):

```
/data/raw/kitti_odometry/
  data_odometry_{color,velodyne,calib,poses}.zip
  dataset/
    sequences/00/{image_2,image_3,velodyne,calib.txt,times.txt}
    sequences/01/...
    poses/00.txt ... 10.txt
```

Convert the evaluation set (static-scene segments of sequences 00-10):

```bash
python scripts/prepare_data/kitti.py \
    --input-dir /data/raw/kitti_odometry --extract \
    --output-dir ${DATA_ROOT}/kitti_static \
    --split-file data/splits/kitti_static/test.txt --workers 8
```

The script defines 33 static segments (time ranges without independently moving
objects); the test split uses 32 of them (all except `03_0m25s_1m20s`). Output
size for the test split is about 55 GB.

Result:

```
kitti_static/
  00_0m11s_1m00s/
    rgb_left/000000.png ...          1240x376 RGB (image_2 cropped to even width)
    rgb_right/000000.png ...         image_3
    depth_left/000000.npy ...        (376, 1241) float32 sparse LiDAR depth, 0 = no data
    depth_right/000000.npy ...
    poses_left.npy                   (N, 4, 4) float32 camera-to-world (left color camera)
    poses_right.npy
    intrinsics_left.npy              (3, 3) float32 (from P0)
    intrinsics_right.npy
    timestamps.npy                   (N,) float64 seconds
```

`--mode full` writes whole sequences (`00/`, `01/`, ...) without depth instead.

## Infinigen stereo video (training + evaluation)

The Infinigen scenes used in the paper are the stereo video sequences of
InfinigenSV, rendered with [Infinigen](https://github.com/princeton-vl/infinigen)
and released on Hugging Face as
[`MatchLab/Infinigen`](https://huggingface.co/datasets/MatchLab/Infinigen)
(Jing et al., "Match Stereo Videos via Bidirectional Alignment"). Our
`infinigen_static` splits use 35 training scenes (30 from the InfinigenSV
`train` folder and 5 from `val`, all without moving objects; about 28 GB) and 9
test scenes (from `test`).

TODO(owner): confirm that the released numbers should reference the public
InfinigenSV scenes (scene IDs in `data/splits/infinigen_static/test.txt` match
InfinigenSV) rather than a separate data release.

Only the PNG images, `.npy` depth maps and `.npz` camera files are needed (skip
the `.exr` images and depth PNGs):

```bash
for s in $(cat data/splits/infinigen_static/test.txt); do
  hf download MatchLab/Infinigen --repo-type dataset \
      --local-dir /data/raw/InfinigenSV \
      --include "test/$s/frames/Image/*/*.png" --include "test/$s/frames/Depth/*/*.npy" \
      --include "test/$s/frames/camview/*/*.npz"
done
# Training scenes (in the train/ or val/ folder):
for s in $(cat data/splits/infinigen_static/train.txt); do
  inc=()
  for d in train val; do
    for p in "Image/*/*.png" "Depth/*/*.npy" "camview/*/*.npz"; do
      inc+=(--include "$d/$s/frames/$p")
    done
  done
  hf download MatchLab/Infinigen --repo-type dataset \
      --local-dir /data/raw/InfinigenSV "${inc[@]}"
done
```

Expected raw tree (standard Infinigen frame export; scene folders are found at
any depth below `--input-dir`):

```
/data/raw/InfinigenSV/
  test/1988b111/frames/
    Image/camera_0/Image_0_0_0001_0.png ...     left
    Image/camera_1/Image_0_0_0001_1.png ...     right
    Depth/camera_{0,1}/Depth_0_0_0001_*.npy ...
    camview/camera_{0,1}/camview_0_0_0001_*.npz ...   'T' (camera-to-world, Blender axes), 'K'
```

Convert:

```bash
python scripts/prepare_data/infinigen.py \
    --input-dir /data/raw/InfinigenSV \
    --output-dir ${DATA_ROOT}/infinigen_processed \
    --split-file data/splits/infinigen_static/test.txt
# and again with data/splits/infinigen_static/train.txt for training
```

Result:

```
infinigen_processed/
  1988b111/
    rgb_left/000000.png ...          1280x720 RGB
    rgb_right/000000.png ...
    depth_left/000000.npy ...        (720, 1280) float32, meters
    depth_right/000000.npy ...
    poses_left.npy                   (N, 4, 4) float32 camera-to-world
    poses_right.npy
    intrinsics_left.npy              (N, 3, 3) float32, per frame
    intrinsics_right.npy
```

Poses: the Blender camera-to-world matrices are left-multiplied by
`diag(1, -1, -1, 1)` (Blender world, Y up, to OpenCV world, Y down).

## XRStereo (training)

Source: [`facebook/xrstereo`](https://huggingface.co/datasets/facebook/xrstereo)
on Hugging Face, the synthetic stereo video dataset introduced with
StereoSplat: 17,922 sequences of 5 rectified 1024x1024 stereo frames with
disparity for both views (72 tar shards, about 373 GB). The loader reads the
released files directly, so no conversion script is needed: download, verify
the checksums and extract the shards (about 373 GB more; delete the tars
afterwards if you need the space):

```bash
hf download facebook/xrstereo --repo-type dataset --local-dir /data/raw/xrstereo
cd /data/raw/xrstereo
python -c "
import hashlib, json
m = json.load(open('manifest.json'))
for s in m['shards']:
    h = hashlib.sha256()
    with open(s['path'], 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    assert h.hexdigest() == s['sha256'], s['path']
print('all', len(m['shards']), 'shards OK')
"
mkdir -p ${DATA_ROOT}/xrstereo
for t in data/*.tar; do tar -xf "$t" -C ${DATA_ROOT}/xrstereo; done
```

Result (`data/splits/xrstereo/train.txt` is a copy of the release's
`splits/train.txt`):

```
xrstereo/sequences/
  seq_000000/
    calibration.json                 {"f", "cx", "cy", "baseline_m"}, both cameras
    poses_left.npy                   (5, 4, 4) float64 camera-to-world, left camera, Blender axes
    frame_0000.npz ... frame_0004.npz
                                     image_left, image_right (1024, 1024, 3) uint8
                                     disparity_left, disparity_right (1024, 1024) float16, pixels
```

The loader (`stereosplat/datasets/xrstereo.py`) converts disparity to metric
depth (`f * baseline_m / disparity`; disparity <= 0 is invalid), converts the
poses to OpenCV axes (`pose @ diag(1, -1, -1, 1)`) and places the right camera
at `+baseline_m` along the left camera's x axis. With `num_views: 3` and
`stride: 1`, each sequence gives one training window (frames 0-2).

---

## Licenses

The scripts are released under this repository's license. The datasets are
**not** redistributed here: you download them from their providers, and you are
responsible for complying with their terms. The training and evaluation code
only reads the converted data.

| Dataset | License (as stated by the providers) |
|---|---|
| TartanAir v1 / v2 | CC BY 4.0 on the project website; the Hugging Face mirrors are tagged BSD-3-Clause. TODO(owner): confirm. |
| KITTI Visual Odometry | Creative Commons Attribution-NonCommercial-ShareAlike 3.0 (CC BY-NC-SA 3.0). Non-commercial use only; data derived from KITTI must be shared under the same license. |
| XRStereo (`facebook/xrstereo`) | Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0). Non-commercial use only. |
| InfinigenSV (`MatchLab/Infinigen`) | Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0). Infinigen code: BSD-3-Clause. |

Please cite the original dataset papers when you use them.
