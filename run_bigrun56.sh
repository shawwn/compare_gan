#!/bin/bash
set -ex
#export TPU_HOST="${TPU_HOST:-10.255.128.3}"
export TPU_HOST="${TPU_HOST:-10.255.128.2}"
export TPU_NAME="${TPU_NAME:-tpu-v3-256-euw4a-28}"
export MODEL_DIR="${MODEL_DIR:-gs://darnbooru-euw4a/runs/bigrun56/}"
export GIN_CONFIG="${GIN_CONFIG:-example_configs/bigrun56.gin}"
export LOG="${LOG:-logs56.txt}"
while true; do
  timeout --signal=SIGKILL 19h python3 wrapper.py compare_gan/main.py --use_tpu --tfds_data_dir 'gs://danbooru-euw4a/tensorflow_datasets/' --model_dir "${MODEL_DIR}" --gin_config "$GIN_CONFIG" "$@" 2>&1 | tee -a "${LOG}"
  sleep 30
done
