# ColRetriever: Representation Learning for Foreign Key Discovery in Large-Scale Data Lakes

ColRetriever is a FK-PK discovery project based on column embeddings.

## Layout

- `model/`: the final demo checkpoint of ColRetriever used for FK-PK retrieval.
  It contains the shared encoder, tokenizer, and FK/PK projection heads.
- `dataset/train/`: small JSONL examples from each training source.
- `dataset/test/`: full test files for `spider`, `BIRD`, and `WikiDBs-test`.
- `script/`: data construction, training, and evaluation scripts.

## Build Training Data

To rebuild the full training data from WikiDBs raw data:

```bash
python3 script/build_train_data.py \
  --wikidbs-root raw_dataset/WikiDBs \
  --output-root dataset/train_full
```

The driver writes WikiDBs FK-PK positives, synthesis positives, col2col hard negatives, nl2col positives, and nl2col hard negatives. You can reduce the targets for a quick smoke run, for example:

```bash
python3 script/build_train_data.py \
  --wikidbs-root raw_dataset/WikiDBs \
  --output-root /tmp/fkpk_demo_train \
  --limit-databases 20 \
  --target-wikidbs 100 \
  --target-synthesis 100 \
  --target-hard 50 \
  --target-nl2col 100 \
  --target-nl2col-hard 50
```

## Train

The local training script uses the normal two-stage logic with a BGE-small
backbone and 12GB-GPU-friendly gradient accumulation. For a real run, point `--data-root` to the full training data, such as `dataset/train` in this
workspace or `dataset/train_full` after running the data builder. The included `dataset/train` directory only contains a few examples for inspection.

```bash
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false python3 script/train_bge_small_two_stage_local.py \
  --data-root dataset/train \
  --output-dir model_trained \
  --run-stage1 \
  --run-stage2 \
  --fp16
```

Default training settings are:

- Stage 1: MLM + synthesis contrastive learning, 6000 steps, LR `2e-5`.
- Stage 2: col2col + nl2col contrastive learning, 10000 steps, LR `1e-5`. Local micro-batch size is 8 with gradient accumulation 8 in this stage.

## Evaluate

Evaluate on WikiDBs-test:

```bash
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false python3 script/evaluate_fk_discovery.py \
  --model-dir model \
  --test-file dataset/test/WikiDBs-test.json \
  --dataset wikidbs \
  --output results/wikidbs_results.json
```

Evaluate on Spider or BIRD:

```bash
python3 script/evaluate_fk_discovery.py \
  --model-dir model \
  --test-file dataset/test/spider.json \
  --dataset spider \
  --output results/spider_results.json
```

For Spider/BIRD with value sampling, keep the original raw dataset directories
available under `raw_dataset/`. If you only want schema-text evaluation, pass
`--sample-size 0 --max-rows 0`.
