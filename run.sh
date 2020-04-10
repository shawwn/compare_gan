#!/bin/bash
set -ex
export TPU_HOST=10.255.128.3
export TPU_NAME=tpu-v3-512-euw4a-27
export MODEL_DIR=gs://danbooru-euw4a/test/bigrun03/
#export DATASETS=gs://danbooru-euw4a/datasets/danbooru2019-s/danbooru2019-s-0*,gs://danbooru-euw4a/datasets/e621-s/e621-s-0*
export DATASETS=gs://danbooru-euw4a/datasets/danbooru2019-s/danbooru2019-s-0*
exec python3 wrapper.py compare_gan/main.py --use_tpu --tfds_data_dir 'gs://danbooru-euw4a/tensorflow_datasets/' --model_dir "${MODEL_DIR}" --gin_config "$@"
