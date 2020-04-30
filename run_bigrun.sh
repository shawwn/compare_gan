#!/bin/bash
set -ex
export TPU_HOST="${TPU_HOST:-10.255.128.3}"
export TPU_NAME="${TPU_NAME:-tpu-v2-128-euw4a-7}"
export MODEL_DIR="${MODEL_DIR:-gs://darnbooru-euw4a/runs/bigrun/}"
export GIN_CONFIG="${GIN_CONFIG:-example_configs/bigrun.gin}"
export TIMEOUT="${TIMEOUT:-19h}"
export LOGFILE="$1"
shift 1
export TENSORFORK_RUN="$1"
shift 1

while true; do
  timeout --signal=SIGKILL "${TIMEOUT}" python3 wrapper.py compare_gan/main.py --use_tpu --tfds_data_dir 'gs://darnbooru-euw4a/tensorflow_datasets/' --model_dir "${MODEL_DIR}" --gin_config "$GIN_CONFIG" "$@" 2>&1 | tee -a "${LOGFILE}"
  sleep 30
done
