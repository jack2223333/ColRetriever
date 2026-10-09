#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}" \
python3 demo/script/evaluate_fk_discovery.py \
  --model-dir demo/model \
  --test-file demo/dataset/test/WikiDBs-test.json \
  --dataset wikidbs \
  --output demo/results/wikidbs_results.json
