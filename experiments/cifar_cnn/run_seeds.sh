#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/cifar_cnn/resnet18-full}"
IFS=',' read -r -a GPU_LIST <<< "${GPUS:-0}"
SEEDS=(42 43 44 45 46)

for index in "${!SEEDS[@]}"; do
  seed="${SEEDS[${index}]}"
  gpu="${GPU_LIST[$((index % ${#GPU_LIST[@]}))]}"
  "${PYTHON}" "${SCRIPT_DIR}/train.py" \
    --dataset cifar100 \
    --arch resnet18 \
    --batch-size 128 \
    --dynamics \
    --save_path "${OUTPUT_DIR}" \
    --manualSeed "${seed}" \
    --gpu "${gpu}" &

  if (( (index + 1) % ${#GPU_LIST[@]} == 0 )); then
    wait
  fi
done

wait
