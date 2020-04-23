#!/bin/bash
set -ex
export TPU_HOST="${TPU_HOST:-10.255.128.3}"
export TPU_NAME="${TPU_NAME:-tpu-v3-128-euw4a-51}"
export MODEL_DIR="${MODEL_DIR:-gs://darnbooru-euw4a/runs/bigrun41/}"
export GIN_CONFIG="${GIN_CONFIG:-example_configs/bigrun41.gin}"
export LOGDIR="${LOGDIR:-logs41.txt}"
while true; do
  python3 wrapper.py compare_gan/main.py --use_tpu --tfds_data_dir 'gs://danbooru-euw4a/tensorflow_datasets/' --model_dir "${MODEL_DIR}" --gin_config "$GIN_CONFIG" "$@" 2>&1 | tee -a "${LOGDIR}"
  sleep 30
done
