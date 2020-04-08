#!/bin/bash
set -ex
export TPU_HOST=10.255.128.3
export TPU_NAME=tpu-v3-256-euw4a-27
export MODEL_DIR=gs://danbooru-euw4a/test/bigrun00/
exec python3 wrapper.py compare_gan/main.py --use_tpu --tfds_data_dir 'gs://danbooru-euw4a/tensorflow_datasets/' --model_dir "${model_dir}" --gin_config "$@"
