#!/bin/sh
set -ex
exec capture_tpu_profile --tpu=tpu-v2-256-euw4a-16 --logdir=gs://danbooru-euw4a/test/bigrun07 --num_tracing_attempts=10 --duration_ms=600000 --gcp_project gpt-2-15b-poetry --tpu_zone europe-west4-a --display_timestamp --tpu_zone europe-west4-a "$@"
