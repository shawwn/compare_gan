#!/bin/bash
set -ex
export PYTHONPATH="$PYTHONPATH:."
#export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/tfk/lib}"
#export TPU_HOST=${TPU_HOST:-10.255.128.3}
unset TPU_HOST
#export TPU_NAME="${TPU_NAME:-tpu-v3-256-euw4a-55}" # RIP big pod
export TPU_NAME="${TPU_NAME:-tpu-v3-128-euw4a-57}"

export RUN_NAME="${RUN_NAME:-bigrun94_big128deep256ch96}"
tmux-set-title "${RUN_NAME} ${TPU_NAME}"
export MODEL_DIR="${MODEL_DIR:-gs://doto-euw4a/runs/bigrun94_big128/dec25/deep256ch96/run8_g_ch128sa64_d_lrmul_0_22}"
export GIN_CONFIG="example_configs/bigrun94_big128deep512ch128.gin"

date="$(python3 -c 'import datetime; print(datetime.datetime.now().strftime("%Y-%m-%d"))')"
logfile="logs/${RUN_NAME}-${date}.txt"
mkdir -p logs

export LABELS=""
export NUM_CLASSES=1000
export TPU_SPLIT_COMPILE_AND_EXECUTE=1
export TF_TPU_WATCHDOG_TIMEOUT=1800
#wrapper=wrapper.py 

# prevent OOMs from killing the training process.
echo -1000 | sudo tee /proc/$$/oom_score_adj

while true; do
  timeout --signal=SIGKILL 16h python3 $wrapper compare_gan/main.py --use_tpu --tfds_data_dir 'gs://dota-euw4a/tensorflow_datasets/' --model_dir "${MODEL_DIR}" --gin_config "$GIN_CONFIG" --gin_bindings "begin_run.model_dir = '${MODEL_DIR}/'" --gin_bindings "begin_run.tpu_name = '${TPU_NAME}'" "$@" 2>&1 | tee -a "$logfile"
  if [ ! -z "$TPU_NO_RECREATE" ]
  then
    echo "Not recreating TPU."
    sleep 30
  else
    echo "Recreating TPU in 30s."
    sleep 30
    # sudo pip3 install -U tpudiepie
    pu recreate "$TPU_NAME" --yes --retry 300
  fi
done
