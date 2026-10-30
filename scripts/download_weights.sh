#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Download the StereoSplat release checkpoint from Hugging Face, verify its
# sha256, and print instructions for the FoundationStereo weights (which are
# distributed by NVIDIA and not redistributed here).
#
# Usage:
#   bash scripts/download_weights.sh [CKPT_ROOT]
# CKPT_ROOT defaults to $CKPT_ROOT or ./checkpoints.

set -euo pipefail

# ---------------------------------------------------------------------------
# TODO(owner): placeholders, set before release.
HF_REPO_ID="facebook/stereo-splat"           # TODO(owner): final Hugging Face repo id
HF_FILENAME="stereosplat_v1.pth"             # TODO(owner): final file name in that repo
HF_REVISION="main"                           # TODO(owner): pin a commit/tag of the HF repo
STEREOSPLAT_SHA256="TODO_OWNER_SHA256_OF_STEREOSPLAT_V1"  # TODO(owner): sha256 of the uploaded file
# Expected sha256 of the current release candidate files (not yet uploaded):
#   stereosplat_v1.pth          3be31e9a9eefffd50a9957ed3b86e471eaa5f13e57085c51b5c8825342665043
#   stereosplat_v1.safetensors  e84556050f65b28367a6bab67606d742db7b4881aa202582d687ad560260184f
# ---------------------------------------------------------------------------

# Public FoundationStereo ViT-S "11-33-40" checkpoint (model_best_bp2.pth).
FS_SHA256="8d7850b9dc68d1366722a02a39745704b1db41471211be6abd93ef463727e6be"

CKPT_ROOT="${1:-${CKPT_ROOT:-checkpoints}}"
mkdir -p "${CKPT_ROOT}"
DEST="${CKPT_ROOT}/${HF_FILENAME}"
URL="https://huggingface.co/${HF_REPO_ID}/resolve/${HF_REVISION}/${HF_FILENAME}"

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

verify() {  # verify <file> <expected sha256>
  local actual
  actual="$(sha256_of "$1")"
  if [[ "$2" == TODO* ]]; then
    echo "WARNING: no reference sha256 configured; got ${actual} for $1"
    return 0
  fi
  if [[ "${actual}" != "$2" ]]; then
    echo "ERROR: sha256 mismatch for $1" >&2
    echo "  expected ${2}" >&2
    echo "  actual   ${actual}" >&2
    return 1
  fi
  echo "OK: sha256 verified for $1"
}

# 1) StereoSplat checkpoint.
if [[ -f "${DEST}" ]]; then
  echo "Found ${DEST}, skipping download."
else
  echo "Downloading ${URL}"
  if command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "${HF_REPO_ID}" "${HF_FILENAME}" \
      --revision "${HF_REVISION}" --local-dir "${CKPT_ROOT}"
  elif command -v curl >/dev/null 2>&1; then
    curl -L --fail -o "${DEST}.part" "${URL}" && mv "${DEST}.part" "${DEST}"
  else
    wget -O "${DEST}.part" "${URL}" && mv "${DEST}.part" "${DEST}"
  fi
fi
verify "${DEST}" "${STEREOSPLAT_SHA256}"

# 2) FoundationStereo weights: manual download from NVIDIA's official links.
FS_DIR="${CKPT_ROOT}/foundation_stereo/11-33-40"
if [[ -f "${FS_DIR}/model_best_bp2.pth" && -f "${FS_DIR}/cfg.yaml" ]]; then
  verify "${FS_DIR}/model_best_bp2.pth" "${FS_SHA256}"
else
  cat <<EOF

FoundationStereo weights not found in ${FS_DIR}.
They are released by NVIDIA under their own license and must be downloaded
from the official links in the FoundationStereo README:

  https://github.com/NVlabs/FoundationStereo

Download the ViT-S model folder "11-33-40" and place its files as:

  ${FS_DIR}/cfg.yaml
  ${FS_DIR}/model_best_bp2.pth   (sha256 ${FS_SHA256})

Then re-run this script to verify the checksum.
EOF
fi
