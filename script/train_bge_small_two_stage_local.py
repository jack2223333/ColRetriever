#!/usr/bin/env python3
"""Two-stage training for BGE column retrieval models.

Stage 1: MLM on serialized table text plus synthesis contrastive learning.
Stage 2: column2column plus nl2column contrastive learning with hard negatives.

Stage 2 uses a shared encoder and two projection heads:
  - FKEncoder: encoder + fk_projection for query/FK/NL side.
  - PKEncoder: encoder + pk_projection for candidate column side.

The column embedding is taken from the hidden state at the target column's
<Column> marker. Natural-language inputs use the first token hidden state.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, IterableDataset
from transformers import (
    AutoModel,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - training can still run without tqdm.
    tqdm = None


SPECIAL_TOKENS = ["<Table>", "<Column>", "<Column-Value>"]
COLUMN_NAME_RE = re.compile(r"<Column>(.*?)(?=<Column-Value>|<Column>|$)")
REPRESENTATION_KEYS = ("column-only", "column+tablename", "schema+context", "column+context")


def format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def progress_bar(total: int, desc: str):
    if tqdm is None or not is_main_process():
        return None
    return tqdm(total=total, desc=desc, dynamic_ncols=True)


def write_timing(output_dir: Path, stage: str, elapsed_seconds: float, extra: dict[str, Any]) -> None:
    if not is_main_process():
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "training_time.json"
    if path.exists():
        timing = json.loads(path.read_text(encoding="utf-8"))
    else:
        timing = {"stages": {}, "total_elapsed_seconds": 0.0}
    timing["stages"][stage] = {
        "elapsed_seconds": elapsed_seconds,
        "elapsed": format_seconds(elapsed_seconds),
        **extra,
    }
    timing["total_elapsed_seconds"] = sum(
        item["elapsed_seconds"] for item in timing["stages"].values()
    )
    timing["total_elapsed"] = format_seconds(timing["total_elapsed_seconds"])
    path.write_text(json.dumps(timing, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def distributed_enabled() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    return rank() == 0


def setup_distributed() -> None:
    if not distributed_enabled():
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed training requires CUDA.")
    torch.cuda.set_device(local_rank())
    dist.init_process_group(backend="nccl")


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def ddp_model(model: nn.Module) -> nn.Module:
    if not distributed_enabled():
        return model
    return DDP(model, device_ids=[local_rank()], output_device=local_rank())


def maybe_no_sync(model: nn.Module, enabled: bool):
    if enabled and isinstance(model, DDP):
        return model.no_sync()
    return nullcontext()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--data-root", type=Path, default=Path("demo/dataset/train"))
    parser.add_argument("--output-dir", type=Path, default=Path("demo/model_trained"))
    parser.add_argument("--run-stage1", action="store_true")
    parser.add_argument("--run-stage2", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--representation", choices=REPRESENTATION_KEYS, default="column+context")
    parser.add_argument(
        "--pooling",
        choices=("auto", "cls", "marker"),
        default="auto",
        help="Column pooling. auto uses the <Column> marker for every column serialization.",
    )
    parser.add_argument(
        "--mlm-sources",
        nargs="+",
        default=["wikidbs", "synthesis", "hard_negatives", "nl2col", "nl2col_hard_negatives"],
    )
    parser.add_argument("--stage1-synthesis-sources", nargs="+", default=["synthesis"])
    parser.add_argument(
        "--positive-sources",
        nargs="+",
        default=["wikidbs"],
        help="Positive col2col source directories.",
    )
    parser.add_argument(
        "--hard-sources",
        nargs="*",
        default=["hard_negatives"],
        help="Hard-negative source directories. Empty disables hard-negative loss.",
    )
    parser.add_argument("--disable-col2col", action="store_true")
    parser.add_argument("--disable-nl2col", action="store_true")
    parser.add_argument("--disable-stage1-synthesis", action="store_true")

    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--mlm-steps", type=int, default=6_000)
    parser.add_argument("--mlm-batch-size", type=int, default=16)
    parser.add_argument("--stage1-synthesis-batch-size", type=int, default=16)
    parser.add_argument("--mlm-grad-accum", type=int, default=2)
    parser.add_argument("--mlm-lr", type=float, default=2e-5)
    parser.add_argument("--mlm-probability", type=float, default=0.15)
    parser.add_argument("--stage1-synthesis-weight", type=float, default=1.0)

    parser.add_argument("--stage2-steps", type=int, default=10_000)
    parser.add_argument("--col-batch-size", type=int, default=8)
    parser.add_argument("--nl-batch-size", type=int, default=8)
    parser.add_argument("--hard-batch-size", type=int, default=8)
    parser.add_argument("--stage2-grad-accum", type=int, default=8)
    parser.add_argument("--stage2-lr", type=float, default=1e-5)
    parser.add_argument("--projection-dim", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--hard-margin", type=float, default=0.2)
    parser.add_argument("--hard-weight", type=float, default=1.0)
    parser.add_argument("--nl-hard-weight", type=float, default=1.0)
    parser.add_argument("--nl-weight", type=float, default=1.0)
    parser.add_argument("--nl-hard-sources", nargs="*", default=["nl2col_hard_negatives"])

    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=1_000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    seed = seed + rank()
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device() -> torch.device:
    if distributed_enabled():
        return torch.device("cuda", local_rank())
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dtype_from_args(args: argparse.Namespace) -> torch.dtype | None:
    if args.bf16:
        return torch.bfloat16
    if args.fp16:
        return torch.float16
    return None


def jsonl_files(root: Path, subdirs: list[str]) -> list[Path]:
    files: list[Path] = []
    for subdir in subdirs:
        files.extend(sorted((root / subdir).glob("*.jsonl")))
    if not files:
        raise FileNotFoundError(f"No JSONL files found under {root} for {subdirs}")
    return files


def cycle_jsonl(files: list[Path], seed: int) -> Iterator[dict[str, Any]]:
    rng = random.Random(seed)
    file_order = list(files)
    while True:
        rng.shuffle(file_order)
        for path in file_order:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        yield json.loads(line)


def add_special_tokens(tokenizer: Any, model: Any) -> None:
    added = tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    if added:
        model.resize_token_embeddings(len(tokenizer))


def table_serialized_texts(record: dict[str, Any], representation: str) -> list[str]:
    if "representationA" in record:
        return [
            record["representationA"][representation],
            record["representationB"][representation],
        ]
    if "column_representations" in record:
        return [record["column_representations"][representation]]
    return []


class MLMDataset(IterableDataset):
    def __init__(self, files: list[Path], seed: int, representation: str) -> None:
        self.files = files
        self.seed = seed
        self.representation = representation

    def __iter__(self) -> Iterator[str]:
        for record in cycle_jsonl(self.files, self.seed):
            for text in table_serialized_texts(record, self.representation):
                yield text


def collate_mlm_texts(tokenizer: Any, max_length: int):
    def collate(texts: list[str]) -> dict[str, torch.Tensor]:
        encoded = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            return_special_tokens_mask=True,
        )
        return [
            {key: encoded[key][index] for key in encoded}
            for index in range(len(texts))
        ]

    return collate


def column_occurrence_index(text: str, target_column: str) -> int:
    target = target_column.strip()
    fallback = 0
    for index, match in enumerate(COLUMN_NAME_RE.finditer(text)):
        name = match.group(1).strip()
        if name == target:
            return index
        if name.lower() == target.lower():
            fallback = index
    return fallback


def contrastive_loss(query: torch.Tensor, candidate: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = query @ candidate.T / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def base_encoder_from_mlm(model: nn.Module) -> nn.Module:
    if hasattr(model, "bert"):
        return model.bert
    if hasattr(model, "base_model"):
        return model.base_model
    raise AttributeError("Cannot find base encoder on MLM model")


def marker_pool_from_encoder(
    encoder: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    marker_id: int,
    occurrence_indices: list[int],
) -> torch.Tensor:
    outputs = encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    hidden = outputs.last_hidden_state
    pooled = []
    for row, occurrence in enumerate(occurrence_indices):
        positions = (input_ids[row] == marker_id).nonzero(as_tuple=False).flatten()
        if len(positions) == 0:
            pooled.append(hidden[row, 0])
        else:
            pos = positions[min(occurrence, len(positions) - 1)]
            pooled.append(hidden[row, pos])
    return torch.stack(pooled, dim=0)


class ProjectionHead(nn.Module):
    def __init__(self, hidden_size: int, projection_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2, dim=-1)


class DualHeadEncoder(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_size: int, projection_dim: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.fk_projection = ProjectionHead(hidden_size, projection_dim)
        self.pk_projection = ProjectionHead(hidden_size, projection_dim)

    def marker_pool(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        marker_id: int,
        occurrence_indices: list[int],
    ) -> torch.Tensor:
        return marker_pool_from_encoder(
            self.encoder, input_ids, attention_mask, marker_id, occurrence_indices
        )

    def cls_pool(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        return outputs.last_hidden_state[:, 0]

    def encode_column(
        self,
        batch: dict[str, torch.Tensor],
        marker_id: int,
        occurrence_indices: list[int],
        head: str,
        pooling: str,
    ) -> torch.Tensor:
        if pooling == "cls":
            pooled = self.cls_pool(batch["input_ids"], batch["attention_mask"])
        else:
            pooled = self.marker_pool(
                batch["input_ids"], batch["attention_mask"], marker_id, occurrence_indices
            )
        return self.fk_projection(pooled) if head == "fk" else self.pk_projection(pooled)

    def encode_text(self, batch: dict[str, torch.Tensor], head: str) -> torch.Tensor:
        pooled = self.cls_pool(batch["input_ids"], batch["attention_mask"])
        return self.fk_projection(pooled) if head == "fk" else self.pk_projection(pooled)

    def forward(
        self,
        kind: str,
        batch: dict[str, torch.Tensor],
        head: str,
        marker_id: int | None = None,
        occurrence_indices: list[int] | None = None,
        pooling: str = "marker",
    ) -> torch.Tensor:
        if kind == "text":
            return self.encode_text(batch, head)
        if kind == "column":
            if marker_id is None or occurrence_indices is None:
                raise ValueError("Column encoding requires marker_id and occurrence_indices.")
            return self.encode_column(batch, marker_id, occurrence_indices, head, pooling)
        raise ValueError(f"Unknown encoding kind: {kind}")


@dataclass
class ColumnBatch:
    texts: list[str]
    columns: list[str]


def take(iterator: Iterator[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    return [next(iterator) for _ in range(n)]


def build_col_positive_batch(
    records: list[dict[str, Any]], representation: str
) -> tuple[ColumnBatch, ColumnBatch]:
    left = ColumnBatch([], [])
    right = ColumnBatch([], [])
    for record in records:
        left.texts.append(record["representationA"][representation])
        left.columns.append(record["columnA"])
        right.texts.append(record["representationB"][representation])
        right.columns.append(record["columnB"])
    return left, right


def build_hard_negative_batch(
    records: list[dict[str, Any]], representation: str
) -> tuple[ColumnBatch, ColumnBatch]:
    return build_col_positive_batch(records, representation)


def build_nl_batch(records: list[dict[str, Any]], representation: str) -> tuple[list[str], ColumnBatch]:
    summaries = []
    columns = ColumnBatch([], [])
    for record in records:
        summaries.append(record["summary"])
        columns.texts.append(record["column_representations"][representation])
        columns.columns.append(record["column"])
    return summaries, columns


def build_nl_hard_batch(records: list[dict[str, Any]], representation: str) -> tuple[list[str], ColumnBatch]:
    summaries = []
    columns = ColumnBatch([], [])
    for record in records:
        summaries.append(record["summary"])
        columns.texts.append(record["column_representations"][representation])
        columns.columns.append(record["column"])
    return summaries, columns


def tokenize_columns(
    tokenizer: Any,
    batch: ColumnBatch,
    max_length: int,
    device_: torch.device,
) -> tuple[dict[str, torch.Tensor], list[int]]:
    occurrence_indices = [
        column_occurrence_index(text, column) for text, column in zip(batch.texts, batch.columns)
    ]
    encoded = tokenizer(
        batch.texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {k: v.to(device_) for k, v in encoded.items()}, occurrence_indices


def tokenize_texts(
    tokenizer: Any,
    texts: list[str],
    max_length: int,
    device_: torch.device,
) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {k: v.to(device_) for k, v in encoded.items()}


def save_stage2(
    path: Path,
    model: nn.Module,
    tokenizer: Any,
    args: argparse.Namespace,
    step: int,
) -> None:
    model = unwrap_model(model)
    path.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(path / "encoder")
    tokenizer.save_pretrained(path / "tokenizer")
    torch.save(
        {
            "fk_projection": model.fk_projection.state_dict(),
            "pk_projection": model.pk_projection.state_dict(),
            "args": vars(args),
            "step": step,
        },
        path / "projection_heads.pt",
    )


def train_stage1(args: argparse.Namespace, tokenizer: Any) -> Path:
    started_at = time.perf_counter()
    stage1_dir = args.output_dir / "stage1_mlm"
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    add_special_tokens(tokenizer, model)
    device_ = device()
    model.to(device_)
    model.train()
    model = ddp_model(model)
    encoder = base_encoder_from_mlm(unwrap_model(model))
    marker_id = tokenizer.convert_tokens_to_ids("<Column>")
    pooling = "marker" if args.pooling == "auto" else args.pooling

    files = jsonl_files(args.data_root, args.mlm_sources)
    dataset = MLMDataset(files, args.seed + rank() * 997, args.representation)
    loader = DataLoader(
        dataset,
        batch_size=args.mlm_batch_size,
        collate_fn=collate_mlm_texts(tokenizer, args.max_length),
        num_workers=args.num_workers,
    )
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_probability
    )
    synthesis_iter = None
    if not args.disable_stage1_synthesis:
        synthesis_files = jsonl_files(args.data_root, args.stage1_synthesis_sources)
        synthesis_iter = cycle_jsonl(synthesis_files, args.seed + 11)

    optimizer = AdamW(model.parameters(), lr=args.mlm_lr)
    updates = args.mlm_steps
    warmup = int(updates * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, updates)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
    autocast_dtype = dtype_from_args(args)

    iterator = iter(loader)
    optimizer.zero_grad(set_to_none=True)
    running = {"loss": 0.0, "mlm": 0.0, "synthesis": 0.0}
    pbar = progress_bar(updates, "stage1")
    for micro_step in range(1, updates * args.mlm_grad_accum + 1):
        batch = collator(next(iterator))
        batch = {k: v.to(device_) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None):
            mlm_loss = model(**batch).loss
            scaled_mlm_loss = mlm_loss / args.mlm_grad_accum
        scaler.scale(scaled_mlm_loss).backward()
        running["loss"] += scaled_mlm_loss.item()
        running["mlm"] += scaled_mlm_loss.item()

        if synthesis_iter is not None and args.stage1_synthesis_batch_size > 0:
            with torch.autocast("cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None):
                synthesis_records = take(synthesis_iter, args.stage1_synthesis_batch_size)
                syn_left, syn_right = build_col_positive_batch(synthesis_records, args.representation)
                syn_l_tok, syn_l_idx = tokenize_columns(tokenizer, syn_left, args.max_length, device_)
                syn_r_tok, syn_r_idx = tokenize_columns(tokenizer, syn_right, args.max_length, device_)
                if pooling == "cls":
                    syn_l_hidden = encoder(
                        input_ids=syn_l_tok["input_ids"],
                        attention_mask=syn_l_tok["attention_mask"],
                        return_dict=True,
                    ).last_hidden_state[:, 0]
                    syn_r_hidden = encoder(
                        input_ids=syn_r_tok["input_ids"],
                        attention_mask=syn_r_tok["attention_mask"],
                        return_dict=True,
                    ).last_hidden_state[:, 0]
                else:
                    syn_l_hidden = marker_pool_from_encoder(
                        encoder,
                        syn_l_tok["input_ids"],
                        syn_l_tok["attention_mask"],
                        marker_id,
                        syn_l_idx,
                    )
                    syn_r_hidden = marker_pool_from_encoder(
                        encoder,
                        syn_r_tok["input_ids"],
                        syn_r_tok["attention_mask"],
                        marker_id,
                        syn_r_idx,
                    )
                syn_q = F.normalize(syn_l_hidden, p=2, dim=-1)
                syn_k = F.normalize(syn_r_hidden, p=2, dim=-1)
                synthesis_loss = contrastive_loss(syn_q, syn_k, args.temperature)
                scaled_synthesis_loss = (
                    args.stage1_synthesis_weight * synthesis_loss / args.mlm_grad_accum
                )
            scaler.scale(scaled_synthesis_loss).backward()
            running["loss"] += scaled_synthesis_loss.item()
            running["synthesis"] += scaled_synthesis_loss.item()

        if micro_step % args.mlm_grad_accum == 0:
            step = micro_step // args.mlm_grad_accum
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(
                    loss=f"{running['loss']:.4f}",
                    mlm=f"{running['mlm']:.4f}",
                    synthesis=f"{running['synthesis']:.4f}",
                )
            if is_main_process() and step % args.log_every == 0:
                print(
                    f"[stage1] step={step} loss={running['loss']:.4f} "
                    f"mlm={running['mlm']:.4f} synthesis={running['synthesis']:.4f}",
                    flush=True,
                )
                running = {"loss": 0.0, "mlm": 0.0, "synthesis": 0.0}
            if is_main_process() and step % args.save_every == 0:
                ckpt = stage1_dir / f"checkpoint-{step}"
                unwrap_model(model).save_pretrained(ckpt)
                tokenizer.save_pretrained(ckpt)

    if pbar is not None:
        pbar.close()
    if is_main_process():
        base_encoder_from_mlm(unwrap_model(model)).save_pretrained(stage1_dir)
        tokenizer.save_pretrained(stage1_dir)
    elapsed = time.perf_counter() - started_at
    write_timing(
        args.output_dir,
        "stage1",
        elapsed,
        {
            "mlm_steps": args.mlm_steps,
            "mlm_grad_accum": args.mlm_grad_accum,
            "stage1_synthesis": not args.disable_stage1_synthesis,
            "representation": args.representation,
        },
    )
    if is_main_process():
        print(f"[stage1] elapsed={format_seconds(elapsed)}", flush=True)
    barrier()
    return stage1_dir


def train_stage2(args: argparse.Namespace, tokenizer: Any, encoder_source: Path | str) -> None:
    started_at = time.perf_counter()
    if args.disable_col2col and args.disable_nl2col:
        raise ValueError("Stage 2 needs col2col, nl2col, or both.")
    encoder = AutoModel.from_pretrained(encoder_source)
    add_special_tokens(tokenizer, encoder)
    hidden_size = encoder.config.hidden_size
    model = DualHeadEncoder(encoder, hidden_size, args.projection_dim).to(device())
    model.train()
    model = ddp_model(model)
    marker_id = tokenizer.convert_tokens_to_ids("<Column>")
    pooling = args.pooling
    if pooling == "auto":
        pooling = "marker"

    positive_iter = None
    hard_iter = None
    nl_iter = None
    nl_hard_iter = None
    if not args.disable_col2col:
        positive_files = jsonl_files(args.data_root, args.positive_sources)
        positive_iter = cycle_jsonl(positive_files, args.seed + 1 + rank() * 997)
        if args.hard_sources:
            hard_files = jsonl_files(args.data_root, args.hard_sources)
            hard_iter = cycle_jsonl(hard_files, args.seed + 2 + rank() * 997)
    if not args.disable_nl2col:
        nl_files = jsonl_files(args.data_root, ["nl2col"])
        nl_iter = cycle_jsonl(nl_files, args.seed + 3 + rank() * 997)
        if args.nl_hard_sources:
            nl_hard_files = jsonl_files(args.data_root, args.nl_hard_sources)
            nl_hard_iter = cycle_jsonl(nl_hard_files, args.seed + 4 + rank() * 997)

    optimizer = AdamW(model.parameters(), lr=args.stage2_lr)
    warmup = int(args.stage2_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, args.stage2_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
    autocast_dtype = dtype_from_args(args)

    optimizer.zero_grad(set_to_none=True)
    running = {"loss": 0.0, "l3": 0.0, "l4": 0.0, "hard": 0.0, "nl_hard": 0.0}
    pbar = progress_bar(args.stage2_steps, "stage2")
    for micro_step in range(1, args.stage2_steps * args.stage2_grad_accum + 1):
        micro_total = 0.0
        micro_l3 = 0.0
        micro_l4 = 0.0
        micro_hard = 0.0
        micro_nl_hard = 0.0

        if positive_iter is not None:
            with torch.autocast("cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None):
                col_records = take(positive_iter, args.col_batch_size)
                col_left, col_right = build_col_positive_batch(col_records, args.representation)
                col_l_tok, col_l_idx = tokenize_columns(tokenizer, col_left, args.max_length, device())
                col_r_tok, col_r_idx = tokenize_columns(tokenizer, col_right, args.max_length, device())
                col_q = model("column", col_l_tok, "fk", marker_id, col_l_idx, pooling)
                col_k = model("column", col_r_tok, "pk", marker_id, col_r_idx, pooling)
                col_loss = contrastive_loss(col_q, col_k, args.temperature)
                scaled_col_loss = col_loss / args.stage2_grad_accum
            scaler.scale(scaled_col_loss).backward()
            micro_total += scaled_col_loss.item()
            micro_l3 += scaled_col_loss.item()

        if hard_iter is not None and args.hard_batch_size > 0:
            with torch.autocast("cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None):
                hard_records = take(hard_iter, args.hard_batch_size)
                hard_left, hard_right = build_hard_negative_batch(hard_records, args.representation)
                hard_l_tok, hard_l_idx = tokenize_columns(tokenizer, hard_left, args.max_length, device())
                hard_r_tok, hard_r_idx = tokenize_columns(tokenizer, hard_right, args.max_length, device())
                hard_q = model("column", hard_l_tok, "fk", marker_id, hard_l_idx, pooling)
                hard_k = model("column", hard_r_tok, "pk", marker_id, hard_r_idx, pooling)
                hard_sim = (hard_q * hard_k).sum(dim=-1)
                hard_loss = F.softplus((hard_sim - args.hard_margin) / args.temperature).mean()
                scaled_hard_loss = args.hard_weight * hard_loss / args.stage2_grad_accum
            scaler.scale(scaled_hard_loss).backward()
            micro_total += scaled_hard_loss.item()
            micro_l3 += scaled_hard_loss.item()
            micro_hard += hard_loss.item() / args.stage2_grad_accum

        if nl_iter is not None:
            with torch.autocast("cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None):
                nl_records = take(nl_iter, args.nl_batch_size)
                nl_texts, nl_columns = build_nl_batch(nl_records, args.representation)
                nl_tok = tokenize_texts(tokenizer, nl_texts, args.max_length, device())
                nl_col_tok, nl_col_idx = tokenize_columns(tokenizer, nl_columns, args.max_length, device())
                nl_q = model("text", nl_tok, "fk")
                nl_k = model("column", nl_col_tok, "pk", marker_id, nl_col_idx, pooling)
                nl_loss = contrastive_loss(nl_q, nl_k, args.temperature)
                scaled_nl_loss = args.nl_weight * nl_loss / args.stage2_grad_accum
            scaler.scale(scaled_nl_loss).backward()
            micro_total += scaled_nl_loss.item()
            micro_l4 += scaled_nl_loss.item()

        if nl_hard_iter is not None and args.hard_batch_size > 0:
            with torch.autocast("cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None):
                nl_hard_records = take(nl_hard_iter, args.hard_batch_size)
                nl_hard_texts, nl_hard_columns = build_nl_hard_batch(
                    nl_hard_records, args.representation
                )
                nl_hard_tok = tokenize_texts(tokenizer, nl_hard_texts, args.max_length, device())
                nl_hard_col_tok, nl_hard_col_idx = tokenize_columns(
                    tokenizer, nl_hard_columns, args.max_length, device()
                )
                nl_hard_q = model("text", nl_hard_tok, "fk")
                nl_hard_k = model(
                    "column", nl_hard_col_tok, "pk", marker_id, nl_hard_col_idx, pooling
                )
                nl_hard_sim = (nl_hard_q * nl_hard_k).sum(dim=-1)
                nl_hard_loss = F.softplus(
                    (nl_hard_sim - args.hard_margin) / args.temperature
                ).mean()
                scaled_nl_hard_loss = args.nl_hard_weight * nl_hard_loss / args.stage2_grad_accum
            scaler.scale(scaled_nl_hard_loss).backward()
            micro_total += scaled_nl_hard_loss.item()
            micro_l4 += scaled_nl_hard_loss.item()
            micro_nl_hard += nl_hard_loss.item() / args.stage2_grad_accum

        running["loss"] += micro_total
        running["l3"] += micro_l3
        running["l4"] += micro_l4
        running["hard"] += micro_hard
        running["nl_hard"] += micro_nl_hard

        if micro_step % args.stage2_grad_accum == 0:
            step = micro_step // args.stage2_grad_accum
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(
                    loss=f"{running['loss']:.4f}",
                    l3=f"{running['l3']:.4f}",
                    l4=f"{running['l4']:.4f}",
                )
            if is_main_process() and step % args.log_every == 0:
                print(
                    f"[stage2] step={step} loss={running['loss']:.4f} "
                    f"l3={running['l3']:.4f} l4={running['l4']:.4f} "
                    f"hard={running['hard']:.4f} nl_hard={running['nl_hard']:.4f}",
                    flush=True,
                )
                running = {"loss": 0.0, "l3": 0.0, "l4": 0.0, "hard": 0.0, "nl_hard": 0.0}
            if is_main_process() and step % args.save_every == 0:
                save_stage2(args.output_dir / "stage2_contrastive" / f"checkpoint-{step}", model, tokenizer, args, step)

    if pbar is not None:
        pbar.close()
    if is_main_process():
        save_stage2(args.output_dir / "stage2_contrastive" / "final", model, tokenizer, args, args.stage2_steps)
    elapsed = time.perf_counter() - started_at
    write_timing(
        args.output_dir,
        "stage2",
        elapsed,
        {
            "stage2_steps": args.stage2_steps,
            "stage2_grad_accum": args.stage2_grad_accum,
            "col2col": not args.disable_col2col,
            "nl2col": not args.disable_nl2col,
            "positive_sources": args.positive_sources,
            "hard_sources": args.hard_sources,
            "nl_hard_sources": args.nl_hard_sources,
            "representation": args.representation,
        },
    )
    if is_main_process():
        print(f"[stage2] elapsed={format_seconds(elapsed)}", flush=True)
    barrier()


def main() -> None:
    setup_distributed()
    args = parse_args()
    try:
        if not args.run_stage1 and not args.run_stage2:
            raise SystemExit("Specify --run-stage1 and/or --run-stage2")
        set_seed(args.seed)
        if is_main_process():
            args.output_dir.mkdir(parents=True, exist_ok=True)
        barrier()

        if is_main_process() and distributed_enabled():
            print(
                f"[ddp] world_size={world_size()} local_rank={local_rank()} "
                f"output_dir={args.output_dir}",
                flush=True,
            )

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

        stage1_dir = args.output_dir / "stage1_mlm"
        if args.run_stage1:
            stage1_dir = train_stage1(args, tokenizer)

        if args.run_stage2:
            encoder_source: Path | str = stage1_dir if stage1_dir.exists() else args.model_name
            tokenizer_source = stage1_dir if stage1_dir.exists() else args.model_name
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
            train_stage2(args, tokenizer, encoder_source)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
