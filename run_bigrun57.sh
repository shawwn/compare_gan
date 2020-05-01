#!/bin/bash
set -ex
#export TPU_HOST="${TPU_HOST:-10.255.128.3}"
export TPU_HOST="${TPU_HOST:-10.255.128.2}"
export TPU_NAME="${TPU_NAME:-tpu-v3-128-euw4a-24}"
export MODEL_DIR="${MODEL_DIR:-gs://darnbooru-euw4a/runs/bigrun57/}"
export GIN_CONFIG="${GIN_CONFIG:-example_configs/bigrun57.gin}"
export LOG="${LOG:-logs57.txt}"
while true; do
  timeout --signal=SIGKILL 19h python3 wrapper.py compare_gan/main.py --use_tpu --tfds_data_dir 'gs://danbooru-euw4a/tensorflow_datasets/' --model_dir "${MODEL_DIR}" --gin_config "$GIN_CONFIG" "$@" 2>&1 | tee -a "${LOG}"
  sleep 30
done
