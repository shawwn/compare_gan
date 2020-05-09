#!/bin/bash
set -ex
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/tfk/lib}"
#export TPU_HOST=10.255.128.3
export TPU_HOST=${TPU_HOST:-10.255.128.2}
export TPU_NAME="${TPU_NAME:-tpu-v3-128-euw4a-29}"
export TENSORFORK_RUN="tensorfork/tensorfork#1"
printf '\033]2;%s\033\\' "bigrun36 ${TPU_NAME} @ ${TPU_HOST} (${TENSORFORK_RUN})" # set tmux title
export MODEL_DIR="${MODEL_DIR:-gs://darnbooru-euw4a/runs/bigrun36c/}"
export DATASETS=gs://danbooru-euw4a/datasets/danbooru2019-s/danbooru2019-s-0*
export LABELS=""
export NUM_CLASSES=1000
export TPU_SPLIT_COMPILE_AND_EXECUTE=1
export TF_TPU_WATCHDOG_TIMEOUT=1800
while true; do
  timeout --signal=SIGKILL 19h python3 wrapper.py compare_gan/main.py --use_tpu --tfds_data_dir 'gs://darnbooru-euw4a/tensorflow_datasets/' --model_dir "${MODEL_DIR}" --gin_config example_configs/biggan_danbooru128.gin "$@" 2>&1 | tee -a logs36.txt
  sleep 30
done
