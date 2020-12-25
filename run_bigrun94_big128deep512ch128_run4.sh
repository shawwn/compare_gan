#!/bin/bash
set -ex
export PYTHONPATH="$PYTHONPATH:."
#export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/tfk/lib}"
#export TPU_HOST=${TPU_HOST:-10.255.128.3}
unset TPU_HOST
#export TPU_NAME="${TPU_NAME:-tpu-v3-256-euw4a-55}" # RIP big pod
export TPU_NAME="${TPU_NAME:-tpu-v3-128-euw4a-56}"

export RUN_NAME="${RUN_NAME:-bigrun94_big128deep256ch96_dec25_deep256ch96_run12_g_ch128sa128_d_lrmul_1_0__bs4}"
tmux-set-title "$${TPU_NAME} ${RUN_NAME}"
export MODEL_DIR="${MODEL_DIR:-gs://doto-euw4a/runs/bigrun94_big128/dec25/deep256ch96/run12_g_ch128sa128_d_lrmul_1_0__bs4}"
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
  timeout --signal=SIGKILL 16h \
    python3 $wrapper compare_gan/main.py \
    --use_tpu \
    --tfds_data_dir 'gs://dota-euw4a/tensorflow_datasets/' \
    --model_dir "${MODEL_DIR}" \
    --gin_config "$GIN_CONFIG" \
    --gin_bindings "begin_run.model_dir = '${MODEL_DIR}/'" \
    --gin_bindings "begin_run.tpu_name = '${TPU_NAME}'" \
    \
    \
    --gin_bindings "conditional_batch_norm.scale_start = 1.0" \
    --gin_bindings "options.batch_per_core = 4" \
    --gin_bindings "spectral_norm_stateless.power_iteration_rounds = 5" \
    --gin_bindings "resnet_biggan_deep.Discriminator.blocks_with_attention = '128'" \
    --gin_bindings "resnet_biggan_deep.Discriminator.ch = 128" \
    --gin_bindings "ModularGAN.d_lr_mul = 1.0" \
    \
    --gin_bindings "stop_loss.enabled = True" \
    --gin_bindings "stop_loss.g_stop_d_above = 1.50" \
    --gin_bindings "stop_loss.g_stop_g_below = None" \
    \
    \
   "$@" 2>&1 | tee -a "$logfile"
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
