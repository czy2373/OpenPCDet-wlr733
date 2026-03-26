#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PARENT_DIR="$(cd "${REPO_ROOT}/.." && pwd)"
STAMP="$(date +%Y%m%d)"
OUTPUT_PATH="${1:-${PARENT_DIR}/autodl_openpcdet_full_dair_alpha_map_${STAMP}.tar.gz}"

echo "Repo root: ${REPO_ROOT}"
echo "Output: ${OUTPUT_PATH}"

cd "${PARENT_DIR}"

tar -czf "${OUTPUT_PATH}" \
  --exclude='OpenPCDet/.git' \
  --exclude='OpenPCDet/data' \
  --exclude='OpenPCDet/output' \
  --exclude='OpenPCDet/tools/output' \
  --exclude='OpenPCDet/build' \
  --exclude='OpenPCDet/dist' \
  --exclude='OpenPCDet/pcdet.egg-info' \
  --exclude='OpenPCDet/__pycache__' \
  --exclude='OpenPCDet/.pytest_cache' \
  --exclude='OpenPCDet/.mypy_cache' \
  --exclude='OpenPCDet/.ruff_cache' \
  --exclude='OpenPCDet/.labelCloud.log' \
  --exclude='OpenPCDet/cuda-keyring_1.1-1_all.deb' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='*.pyo' \
  --exclude='*.log' \
  --exclude='*.pth' \
  --exclude='*.pt' \
  --exclude='*.swp' \
  --exclude='*.bak' \
  --exclude='*.tmp' \
  --exclude='*.Zone.Identifier' \
  --exclude='OpenPCDet/*.log' \
  --exclude='OpenPCDet/**/*.log' \
  --exclude='OpenPCDet/**/*.pyc' \
  --exclude='OpenPCDet/**/*.pyo' \
  --exclude='OpenPCDet/**/*.pth' \
  --exclude='OpenPCDet/**/*.pt' \
  --exclude='OpenPCDet/**/*.swp' \
  --exclude='OpenPCDet/**/*.bak' \
  --exclude='OpenPCDet/**/*.tmp' \
  --exclude='OpenPCDet/**/*.Zone.Identifier' \
  OpenPCDet

echo "Created package:"
ls -lh "${OUTPUT_PATH}"
