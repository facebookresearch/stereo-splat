# External (MAC-VO) poses for evaluation

In the paper, baselines are evaluated with ground-truth poses, while StereoSplat
is evaluated with camera poses estimated by the stereo visual odometry method
[MAC-VO](https://github.com/MAC-VO/MAC-VO) (Qiu et al., "MAC-VO: Metrics-aware
Covariance for Learning-based Stereo Visual Odometry"). This page describes the
pose files the evaluation expects and how to produce them with upstream MAC-VO.

Precomputed pose files for the test splits are not included in this release
(pending approval). TODO(owner): link them here if they are released.

## What the evaluation reads

```
<poses_dir>/<dataset>/<sequence>/<pose_type>.npy
```

* `<dataset>` is the dataset key in the eval config: `tartanv2`, `kitti`,
  `infinigen`.
* `<sequence>` is the processed sequence folder name, e.g.
  `AbandonedFactory2_hard_P004`, `00_0m11s_1m00s`, `1988b111`.
* `<pose_type>` defaults to `poses_left_aligned_scaled` (the setting used for
  all paper numbers).
* Content: `(N, 4, 4)` float32 camera-to-world matrices of the **left** camera,
  OpenCV axes (x right, y down, z forward), one pose per processed frame, in the
  same order and with the same count as `rgb_left/` (`rgb_lcam_front/` for
  TartanAir v2).

How the poses are used, per evaluation window (3 stereo pairs, stride 1):

1. The left poses of the window's frames are selected and normalized so that the
   first one is the identity (`c2w_i <- inv(c2w_0) @ c2w_i`), then inverted to
   world-to-camera. Consequently only relative poses matter; any global rigid
   transform of the trajectory has no effect, but its **scale** does.
2. The right camera is not read from disk: its pose is the estimated left pose
   composed with the ground-truth left-to-right stereo extrinsics of the dataset.

The files `poses_right*.npy` written by the script below are therefore optional
and unused by the evaluation.

## Pose variants

For every sequence the conversion script writes three left-camera variants:

| File | Meaning |
|---|---|
| `poses_left.npy` | Raw MAC-VO trajectory (first frame at the first GT pose; metric scale from the stereo baseline). |
| `poses_left_aligned.npy` | SE(3)-aligned to GT (rotation + translation, no scale). |
| `poses_left_aligned_scaled.npy` | Sim(3)-aligned to GT (rotation + translation + one global scale). **Used in the paper.** |

Alignment details (`scripts/align_external_poses.py`, identical to the procedure
used for the paper):

1. GT left poses of the whole sequence are normalized so frame 0 is the identity.
2. The MAC-VO trajectory (OpenCV axes, frame 0 at the identity) is aligned to the
   normalized GT camera centers with the Umeyama method, once without scale
   (SE3) and once with scale (Sim3). The alignment uses all frames of the
   sequence.
3. The transform is applied to full poses (`R_al = R @ R_est`,
   `t_al = s * R @ t_est + t`) and expressed in the GT world frame by
   left-multiplying the first GT pose.

Note that the Sim(3) variant uses one GT-derived scale factor per sequence (the
standard protocol for evaluating trajectories); within a sequence, all relative
motion comes from MAC-VO. `poses_left_aligned` keeps MAC-VO's own metric scale.

## Producing the poses with upstream MAC-VO

1. **Prepare the test data** with `scripts/prepare_data/` (see its README). The
   poses must be computed on the processed sequences so that frame indices
   match.

2. **Install MAC-VO** following the upstream README
   (https://github.com/MAC-VO/MAC-VO) and download its pretrained frontend
   weights (`MACVO_FrontendCov.pth`, the FlowFormer-based frontend with
   covariance head; the paper poses used this checkpoint).
   TODO(owner): pin the upstream commit and confirm the weight file name/URL.

3. **Run MAC-VO on every test sequence.** Inputs used for the paper:
   * left/right images from `rgb_left/`, `rgb_right/` (TartanAir v2:
     `rgb_lcam_front/`, `rgb_rcam_front/`), all frames, in order;
   * images resized and center-cropped to 432x960 (H x W), the frontend's
     training resolution: `scale = max(432 / H, 960 / W)`, resize to
     `(int(W * scale), int(H * scale))` (Lanczos when downscaling, bilinear
     when upscaling), then center crop;
   * intrinsics from `intrinsics_left.npy` (first frame), scaled by `scale` and
     shifted by the crop offset;
   * stereo baseline = distance between the first left and right GT camera
     centers (`poses_left.npy[0]`, `poses_right.npy[0]`);
   * the trajectory starts at the identity at frame 0.

   Upstream MAC-VO is configured through an odometry config and a sequence
   (data) config. TODO(owner): add the exact command and a sequence config / data
   loader for the processed layout, e.g.

   ```bash
   # TODO(owner): confirm script name, flags and config files against upstream MAC-VO.
   python MACVO.py --odom Config/Experiment/MACVO/MACVO.yaml \
       --data <sequence config pointing at one processed sequence>
   ```

   MAC-VO reports camera-to-world poses in a NED world frame. Save the
   trajectory as an `(N, 7)` array `tx ty tz qx qy qz qw` (NED, MAC-VO's native
   output; TODO(owner): confirm the output file name and layout) or as `(N, 4, 4)`
   matrices.

   Note: the paper poses were computed with a re-packaged inference-only copy of
   MAC-VO using the same frontend weights and the inputs above. Upstream MAC-VO
   should produce equivalent trajectories, but small numerical differences (and
   therefore small metric differences) are possible.

4. **Align and write the pose files:**

   ```bash
   DATASET=tartanv2                 # or kitti, infinigen
   PROC=${DATA_ROOT}/tartanair_v2_processed     # kitti: kitti_static, infinigen: infinigen_processed
   L=poses_lcam_front.npy; R=poses_rcam_front.npy   # kitti/infinigen: poses_left.npy / poses_right.npy
   for SEQ in $(cat data/splits/tartanv2/test.txt); do   # kitti_static / infinigen_static splits
     python scripts/align_external_poses.py \
         --est macvo_results/$DATASET/$SEQ/poses.npy --est-format ned7 \
         --gt-left $PROC/$SEQ/$L --gt-right $PROC/$SEQ/$R \
         --output-dir ${DATA_ROOT}/macvo_poses/$DATASET/$SEQ
   done
   ```

   `--est-format` accepts `ned7` (`(N, 7)` NED), `ned44` (`(N, 4, 4)` NED) and
   `cv44` (`(N, 4, 4)` OpenCV). The script converts NED to OpenCV with
   `T_cv = P @ T_ned @ P^T`, where `P` maps NED (forward, right, down)
   coordinates to OpenCV (right, down, forward). It also writes `metrics.json`
   with the ATE (RMSE, meters) before alignment, after SE3 and after Sim3
   alignment, which is a useful sanity check (Sim3 ATE should be well below
   1 m on these sequences).

5. **Evaluate:**

   ```bash
   python scripts/evaluate.py --config configs/eval_tartanv2.yaml \
       --checkpoint ${CKPT_ROOT}/stereosplat_v1.pth \
       --poses-dir ${DATA_ROOT}/macvo_poses \
       --pose-type poses_left_aligned_scaled \
       --output-dir outputs/eval_tartanv2
   ```

   The evaluation reports both GT-pose and external-pose results for every test
   dataset.
