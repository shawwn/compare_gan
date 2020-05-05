#!/bin/bash
set -ex
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/tfk/lib}"
export TPU_HOST="${TPU_HOST:-10.255.128.3}"
export TPU_NAME="${TPU_NAME:-tpu-v3-128-euw4a-50}"
printf '\033]2;%s\033\\' "bigrun40 ${TPU_NAME}" # set tmux title
export MODEL_DIR="${MODEL_DIR:-gs://darnbooru-euw4a/runs/bigrun40/}"
export GIN_CONFIG="${GIN_CONFIG:-example_configs/bigrun40.gin}"
export LOGDIR="${LOGDIR:-logs40.txt}"
export TPU_SPLIT_COMPILE_AND_EXECUTE=1
export TF_TPU_WATCHDOG_TIMEOUT=1800
while true; do
  timeout --signal=SIGKILL 19h python3 wrapper.py compare_gan/main.py --use_tpu --tfds_data_dir 'gs://danbooru-euw4a/tensorflow_datasets/' --model_dir "${MODEL_DIR}" --gin_config "$GIN_CONFIG" "$@" 2>&1 | tee -a "${LOGDIR}"
  sleep 30
done
