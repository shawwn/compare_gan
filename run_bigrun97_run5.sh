#!/bin/bash
set -ex
export PYTHONPATH="$PYTHONPATH:."
#export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/tfk/lib}"
#export TPU_HOST=${TPU_HOST:-10.255.128.3}
unset TPU_HOST
export TPU_NAME="${TPU_NAME:-tpu-v3-8-euw4a-0}"

export RUN_NAME="${RUN_NAME:-bigrun97_dec28_run5_evos0_danbooru_dlr_mul_0_2}"
tmux-set-title "$${TPU_NAME} ${RUN_NAME}"
export MODEL_DIR="${MODEL_DIR:-gs://mlpublic-euw4/runs/bigrun97/dec28/run5_evos0_danbooru_dlr_mul_0_2}"
export GIN_CONFIG="example_configs/bigrun97.gin"

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
    --tfds_data_dir 'gs://ml-euw4/tensorflow_datasets/' \
    --model_dir "${MODEL_DIR}" \
    --gin_config "$GIN_CONFIG" \
    --gin_bindings "begin_run.model_dir = '${MODEL_DIR}/'" \
    --gin_bindings "begin_run.tpu_name = '${TPU_NAME}'" \
    \
    --gin_bindings "standardize_batch.use_evonorm = True" \
    --gin_bindings "options.bn_activation = 'auto'" \
    \
    --gin_bindings "conditional_batch_norm.scale_start = 1.0" \
    --gin_bindings "options.batch_per_core = 4" \
    --gin_bindings "ModularGAN.d_lr_mul = 1.0" \
    \
    --gin_bindings "options.datasets = 'gs://mldata-euw4/datasets/danbooru2019-s/danbooru2019-s-0*'" \
    --gin_bindings "dataset_parser.label_bias = 0" \
    --gin_bindings "options.random_labels = True" \
    \
    --gin_bindings "flood_loss.enabled = False" \
    --gin_bindings "options.d_flood =  0.20" \
    --gin_bindings "options.g_flood = -0.40" \
    \
    --gin_bindings "stop_loss.enabled = True" \
    --gin_bindings "stop_loss.d_stop_d_below = 0.20" \
    --gin_bindings "stop_loss.g_stop_d_above = None" \
    --gin_bindings "stop_loss.g_stop_g_below = None" \
    --gin_bindings "stop_loss.d_stop_g_above = None" \
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
