#!/bin/bash
set -ex
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/tfk/lib}"
export TPU_HOST=${TPU_HOST:-10.255.128.3}
export TPU_NAME="${TPU_NAME:-tpu-v3-128-euw4a-13}"

export RUN_NAME="bigrun87"
tmux-set-title "${RUN_NAME} ${TPU_NAME}"
export MODEL_DIR="${MODEL_DIR:-gs://dota-euw4a/runs/bigrun87d/}"
export GIN_CONFIG="example_configs/bigrun87.gin"

date="$(python3 -c 'import datetime; print(datetime.datetime.now().strftime("%Y-%m-%d"))')"
logfile="logs/${RUN_NAME}-${date}.txt"
mkdir -p logs

export LABELS=""
export NUM_CLASSES=1000
export TPU_SPLIT_COMPILE_AND_EXECUTE=1
export TF_TPU_WATCHDOG_TIMEOUT=1800


while true; do
  timeout --signal=SIGKILL 19h python3 wrapper.py compare_gan/main.py --use_tpu --tfds_data_dir 'gs://dota-euw4a/tensorflow_datasets/' --model_dir "${MODEL_DIR}" --gin_config "$GIN_CONFIG" "$@" 2>&1 | tee -a "$logfile"
  if [ ! -z "$TPU_NO_RECREATE" ]
  then
    echo "Not recreating TPU."
    sleep 30
  else
    echo "Recreating TPU in 30s."
    sleep 30
    # sudo pip3 install -U tpudiepie
    pu recreate "$TPU_NAME" --yes
  fi
done
