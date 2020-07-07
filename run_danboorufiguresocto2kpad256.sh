#!/bin/bash
set -ex
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/tfk/lib}"
export TPU_HOST=${TPU_HOST:-10.255.128.3}
export TPU_NAME="${TPU_NAME:-tpu-v3-256-euw4a-6}"
tmux-set-title "bigrun85 ${TPU_NAME}"
export MODEL_DIR="${MODEL_DIR:-gs://dota-euw4a/runs/bigrun85/}"
export LABELS=""
export NUM_CLASSES=1000
export TPU_SPLIT_COMPILE_AND_EXECUTE=1
export TF_TPU_WATCHDOG_TIMEOUT=1800
while true; do
  timeout --signal=SIGKILL 19h python3 wrapper.py compare_gan/main.py --use_tpu --tfds_data_dir 'gs://dota-euw4a/tensorflow_datasets/' --model_dir "${MODEL_DIR}" --gin_config example_configs/biggan_danboorufiguresocto2kpad256.gin "$@" 2>&1 | tee -a logs85.txt
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

