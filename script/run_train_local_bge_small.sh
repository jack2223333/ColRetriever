#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}" \
python3 demo/script/train_bge_small_two_stage_local.py \
  --data-root "${DATA_ROOT:-dataset/train}" \
  --output-dir "${OUTPUT_DIR:-demo/model_trained}" \
  --run-stage1 \
  --run-stage2 \
  --fp16
