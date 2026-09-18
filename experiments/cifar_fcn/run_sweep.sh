#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SEED="${SEED:-0}"
HIDDEN_DIM="${HIDDEN_DIM:-512}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR_ADAM="${LR_ADAM:-1e-3}"
LR_SGD="${LR_SGD:-1e-2}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/cifar_fcn}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data}"
PYTHON="${PYTHON:-python3}"

INITS=(gaussian identity)
ACTIVATIONS=(linear relu)
DEPTHS=(2 3 5 7)
OPTIMIZERS=(adam sgd)
IFS=',' read -r -a GPU_LIST <<< "${GPUS:-0}"

mkdir -p "${OUTPUT_DIR}"

job_index=0
for optimizer in "${OPTIMIZERS[@]}"; do
  for init_type in "${INITS[@]}"; do
    for activation in "${ACTIVATIONS[@]}"; do
      for depth in "${DEPTHS[@]}"; do
        gpu="${GPU_LIST[$((job_index % ${#GPU_LIST[@]}))]}"
        if [[ "${optimizer}" == "adam" ]]; then
          learning_rate="${LR_ADAM}"
        else
          learning_rate="${LR_SGD}"
        fi

        prefix="${OUTPUT_DIR}/${optimizer}_${init_type}_${activation}_depth${depth}_seed${SEED}"
        echo "Launching optimizer=${optimizer}, init=${init_type}, activation=${activation}, depth=${depth}, seed=${SEED}, gpu=${gpu}"

        "${PYTHON}" "${SCRIPT_DIR}/train.py" \
          --gpu "${gpu}" \
          --depth "${depth}" \
          --hidden-dim "${HIDDEN_DIM}" \
          --epochs "${EPOCHS}" \
          --batch-size "${BATCH_SIZE}" \
          --lr "${learning_rate}" \
          --optimizer "${optimizer}" \
          --init-type "${init_type}" \
          --activation "${activation}" \
          --seed "${SEED}" \
          --data-dir "${DATA_DIR}" \
          --out-json "${prefix}.json" \
          --out-csv "${OUTPUT_DIR}/all_results.csv" &

        job_index=$((job_index + 1))
        if (( job_index % ${#GPU_LIST[@]} == 0 )); then
          wait
        fi
      done
    done
  done
done

wait
echo "All experiments finished."
