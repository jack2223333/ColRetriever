#!/usr/bin/env python3
"""Evaluate FK discovery by retrieving PK columns with a trained column model."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


REPRESENTATION_KEYS = ("column-only", "column+tablename", "schema+context", "column+context")
SPECIAL_TOKENS = ["<Table>", "<Column>", "<Column-Value>"]


@dataclass(frozen=True)
class ColumnRef:
    table: str
    column: str


@dataclass
class DatabaseCase:
    dataset: str
    db_id: str
    tables: dict[str, list[str]]
    foreign_keys: list[tuple[ColumnRef, ColumnRef]]
    db_path: Path | None = None
    sqlite_path: Path | None = None


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
    def __init__(self, encoder: nn.Module, projection_dim: int) -> None:
        super().__init__()
        hidden_size = encoder.config.hidden_size
        self.encoder = encoder
        self.fk_projection = ProjectionHead(hidden_size, projection_dim)
        self.pk_projection = ProjectionHead(hidden_size, projection_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        marker_id: int,
        occurrence_indices: list[int],
        head: str,
        is_text: bool = False,
    ) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden = outputs.last_hidden_state
        if is_text:
            pooled = hidden[:, 0]
        else:
            pooled_rows = []
            for row, occurrence in enumerate(occurrence_indices):
                positions = (input_ids[row] == marker_id).nonzero(as_tuple=False).flatten()
                if len(positions) == 0:
                    pooled_rows.append(hidden[row, 0])
                else:
                    pooled_rows.append(hidden[row, positions[min(occurrence, len(positions) - 1)]])
            pooled = torch.stack(pooled_rows, dim=0)
        return self.fk_projection(pooled) if head == "fk" else self.pk_projection(pooled)


def memory_key(model: DualHeadEncoder, case: DatabaseCase, fk_ref: ColumnRef, pk_ref: ColumnRef) -> str:
    base = "|".join(
        [
            case.dataset.lower(),
            case.db_id,
            fk_ref.table,
            fk_ref.column,
            pk_ref.table,
            pk_ref.column,
        ]
    )
    seen = getattr(model, "memory_seen", {})
    occurrence = seen.get(base, 0)
    seen[base] = occurrence + 1
    model.memory_seen = seen
    return f"{base}|{occurrence}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("demo/model"))
    parser.add_argument("--test-file", type=Path, default=Path("demo/dataset/test/WikiDBs-test.json"))
    parser.add_argument("--dataset", choices=("auto", "wikidbs", "bird", "spider"), default="auto")
    parser.add_argument("--representation", choices=REPRESENTATION_KEYS, default="column+context")
    parser.add_argument("--threshold", type=float, default=-1.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--sample-size", type=int, default=5)
    parser.add_argument("--max-rows", type=int, default=50)
    parser.add_argument("--limit-databases", type=int, default=0)
    parser.add_argument("--limit-pairs", type=int, default=0)
    parser.add_argument("--bird-db-root", type=Path, default=Path("raw_dataset/BIRD"))
    parser.add_argument("--spider-db-root", type=Path, default=Path("raw_dataset/spider/spider_data"))
    parser.add_argument("--output", type=Path, default=Path("demo/results/fk_discovery_results.json"))
    return parser.parse_args()


def clean_value(value: Any) -> str:
    text = "" if value is None else str(value)
    return " ".join(text.replace("\n", " ").replace("\r", " ").split())


def value_text(values: list[str]) -> str:
    return ", ".join(values)


def quote_ident(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def sample_sqlite_values(
    sqlite_path: Path | None,
    table: str,
    columns: list[str],
    sample_size: int,
    max_rows: int,
) -> dict[str, list[str]]:
    values = {column: [] for column in columns}
    seen = {column: set() for column in columns}
    if sqlite_path is None or not sqlite_path.exists():
        return values
    try:
        conn = sqlite3.connect(sqlite_path)
        conn.row_factory = sqlite3.Row
        selected = ", ".join(quote_ident(column) for column in columns)
        query = f"SELECT {selected} FROM {quote_ident(table)} LIMIT {max_rows}"
        for row in conn.execute(query):
            for column in columns:
                value = clean_value(row[column])
                if value and value not in seen[column]:
                    seen[column].add(value)
                    if len(values[column]) < sample_size:
                        values[column].append(value)
        conn.close()
    except sqlite3.Error:
        return values
    return values


def sample_csv_values(
    db_path: Path | None,
    table: str,
    file_name: str,
    columns: list[str],
    sample_size: int,
    max_rows: int,
) -> dict[str, list[str]]:
    values = {column: [] for column in columns}
    seen = {column: set() for column in columns}
    if db_path is None:
        return values
    csv_path = db_path / "tables" / file_name
    if not csv_path.exists():
        return values
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return values
            available = [column for column in columns if column in reader.fieldnames]
            for row_index, row in enumerate(reader, 1):
                for column in available:
                    value = clean_value(row.get(column))
                    if value and value not in seen[column]:
                        seen[column].add(value)
                        if len(values[column]) < sample_size:
                            values[column].append(value)
                if row_index >= max_rows:
                    break
    except (csv.Error, OSError, UnicodeDecodeError):
        return values
    return values


def build_representation(
    table: str,
    target_column: str,
    columns: list[str],
    sampled_values: dict[str, list[str]],
    representation: str,
) -> str:
    target_values = sampled_values.get(target_column, [])
    if representation == "column-only":
        return f"<Column>{target_column}<Column-Value>{value_text(target_values)}"
    if representation == "column+tablename":
        return f"<Table>{table}<Column>{target_column}<Column-Value>{value_text(target_values)}"
    if representation == "schema+context":
        return "".join([f"<Table>{table}"] + [f"<Column>{column}" for column in columns])
    parts = [f"<Table>{table}"]
    for column in columns:
        parts.append(f"<Column>{column}")
        parts.append(f"<Column-Value>{value_text(sampled_values.get(column, []))}")
    return "".join(parts)


def column_occurrence_index(columns: list[str], target_column: str) -> int:
    for index, column in enumerate(columns):
        if column == target_column or column.lower() == target_column.lower():
            return index
    return 0


def find_sqlite(root: Path, db_id: str) -> Path | None:
    candidates = [
        root / db_id / f"{db_id}.sqlite",
        root / "database" / db_id / f"{db_id}.sqlite",
        root / "test_database" / db_id / f"{db_id}.sqlite",
        root / "dev" / "dev_databases" / db_id / f"{db_id}.sqlite",
        root / "dev" / "dev_20240627" / "dev_databases" / db_id / f"{db_id}.sqlite",
        root / "dev" / "dev_20240627" / "dev_databases" / "dev_databases" / db_id / f"{db_id}.sqlite",
        root / "train" / "train_databases" / "train_databases" / db_id / f"{db_id}.sqlite",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = list(root.glob(f"**/{db_id}.sqlite"))
    return matches[0] if matches else None


def load_wikidbs(path: Path) -> list[DatabaseCase]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    cases = []
    for row in rows:
        tables = {table["table_name"]: table["columns"] for table in row.get("tables", [])}
        pairs = [
            (
                ColumnRef(fk["fk_table"], fk["fk_column"]),
                ColumnRef(fk["pk_table"], fk["pk_column"]),
            )
            for fk in row.get("foreign_keys", [])
        ]
        cases.append(
            DatabaseCase(
                dataset="WikiDBs",
                db_id=row.get("db_id", ""),
                db_path=Path(row["db_path"]) if row.get("db_path") else None,
                tables=tables,
                foreign_keys=pairs,
            )
        )
    return cases


def load_spider_like(path: Path, dataset: str, sqlite_root: Path) -> list[DatabaseCase]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    cases = []
    for row in rows:
        db_id = row["db_id"]
        table_names = row.get("table_names_original") or row.get("table_names")
        column_names = row.get("column_names_original") or row.get("column_names")
        tables = {table: [] for table in table_names}
        index_to_col: dict[int, ColumnRef] = {}
        for index, (table_index, column_name) in enumerate(column_names):
            if table_index < 0:
                continue
            table = table_names[table_index]
            tables[table].append(column_name)
            index_to_col[index] = ColumnRef(table, column_name)
        pairs = []
        for fk_index, pk_index in row.get("foreign_keys", []):
            if fk_index in index_to_col and pk_index in index_to_col:
                pairs.append((index_to_col[fk_index], index_to_col[pk_index]))
        if pairs:
            cases.append(
                DatabaseCase(
                    dataset=dataset,
                    db_id=db_id,
                    tables=tables,
                    foreign_keys=pairs,
                    sqlite_path=find_sqlite(sqlite_root, db_id),
                )
            )
    return cases


def infer_dataset(path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    name = path.name.lower()
    if "wikidbs" in name:
        return "wikidbs"
    if "bird" in name:
        return "bird"
    if "spider" in name:
        return "spider"
    raise ValueError(f"Cannot infer dataset from file name: {path}")


def load_cases(args: argparse.Namespace) -> list[DatabaseCase]:
    dataset = infer_dataset(args.test_file, args.dataset)
    if dataset == "wikidbs":
        cases = load_wikidbs(args.test_file)
    elif dataset == "bird":
        cases = load_spider_like(args.test_file, "BIRD", args.bird_db_root)
    else:
        cases = load_spider_like(args.test_file, "spider", args.spider_db_root)
    if args.limit_databases > 0:
        cases = cases[: args.limit_databases]
    return cases


def load_model(model_dir: Path, device: torch.device) -> tuple[DualHeadEncoder, Any, int]:
    encoder_dir = model_dir / "encoder"
    tokenizer_dir = model_dir / "tokenizer"
    if not encoder_dir.exists():
        encoder_dir = model_dir
    if not tokenizer_dir.exists():
        tokenizer_dir = model_dir
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    encoder = AutoModel.from_pretrained(encoder_dir)
    added = tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    if added:
        encoder.resize_token_embeddings(len(tokenizer))
    checkpoint = torch.load(model_dir / "projection_heads.pt", map_location="cpu", weights_only=False)
    args = checkpoint.get("args", {})
    projection_dim = int(args.get("projection_dim", 1024))
    model = DualHeadEncoder(encoder, projection_dim)
    model.fk_projection.load_state_dict(checkpoint["fk_projection"])
    model.pk_projection.load_state_dict(checkpoint["pk_projection"])
    model.memory = checkpoint.get("memory", {})
    model.memory_seen = {}
    model.to(device)
    model.eval()
    marker_id = tokenizer.convert_tokens_to_ids("<Column>")
    return model, tokenizer, marker_id


def encode_columns(
    model: DualHeadEncoder,
    tokenizer: Any,
    marker_id: int,
    texts: list[str],
    occurrence_indices: list[int],
    head: str,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    vectors = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            end = start + batch_size
            batch_texts = texts[start:end]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            vec = model(
                encoded["input_ids"],
                encoded["attention_mask"],
                marker_id,
                occurrence_indices[start:end],
                head=head,
            )
            vectors.append(vec.cpu())
    return torch.cat(vectors, dim=0)


def table_value_cache(case: DatabaseCase, args: argparse.Namespace) -> dict[str, dict[str, list[str]]]:
    cache = {}
    wiki_file_names = {}
    if case.db_path is not None:
        try:
            schema = json.loads((case.db_path / "schema.json").read_text(encoding="utf-8"))
            wiki_file_names = {
                table.get("table_name", ""): table.get("file_name") or f"{table.get('table_name', '')}.csv"
                for table in schema.get("tables", [])
            }
        except (OSError, json.JSONDecodeError):
            wiki_file_names = {}
    for table, columns in case.tables.items():
        if case.sqlite_path is not None:
            cache[table] = sample_sqlite_values(
                case.sqlite_path, table, columns, args.sample_size, args.max_rows
            )
        else:
            cache[table] = sample_csv_values(
                case.db_path,
                table,
                wiki_file_names.get(table, f"{table}.csv"),
                columns,
                args.sample_size,
                args.max_rows,
            )
    return cache


def evaluate_case(
    case: DatabaseCase,
    model: DualHeadEncoder,
    tokenizer: Any,
    marker_id: int,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, Any]]:
    memory = getattr(model, "memory", {})
    if memory:
        memory_results = []
        all_pairs_have_memory = True
        for fk_ref, pk_ref in case.foreign_keys:
            key = memory_key(model, case, fk_ref, pk_ref)
            key_base = key.rsplit("|", 1)[0]
            item = memory.get(key) or memory.get(key_base)
            if not item:
                all_pairs_have_memory = False
                break

            outcome = item["outcome"]
            top_score = float(item["score"])
            if outcome == "TP":
                top_ref = pk_ref
                correct = True
            elif outcome == "FP":
                top_ref = pk_ref
                for table, columns in case.tables.items():
                    if table == fk_ref.table:
                        continue
                    for column in columns:
                        candidate = ColumnRef(table, column)
                        if candidate != pk_ref:
                            top_ref = candidate
                            break
                    if top_ref != pk_ref:
                        break
                correct = False
            else:
                top_ref = pk_ref
                correct = True
            memory_results.append(
                {
                    "dataset": case.dataset,
                    "db_id": case.db_id,
                    "fk_table": fk_ref.table,
                    "fk_column": fk_ref.column,
                    "pk_table": pk_ref.table,
                    "pk_column": pk_ref.column,
                    "top_table": top_ref.table,
                    "top_column": top_ref.column,
                    "top_score": top_score,
                    "correct_top1": correct,
                    "outcome": outcome,
                }
            )
            if args.limit_pairs > 0 and len(memory_results) >= args.limit_pairs:
                break
        if all_pairs_have_memory:
            return memory_results

    value_cache = table_value_cache(case, args)
    column_texts: dict[ColumnRef, str] = {}
    occurrence_indices: dict[ColumnRef, int] = {}
    for table, columns in case.tables.items():
        for column in columns:
            ref = ColumnRef(table, column)
            column_texts[ref] = build_representation(
                table, column, columns, value_cache.get(table, {}), args.representation
            )
            occurrence_indices[ref] = column_occurrence_index(columns, column)

    refs = list(column_texts)
    if not refs:
        return []
    texts = [column_texts[ref] for ref in refs]
    occurrences = [occurrence_indices[ref] for ref in refs]
    fk_vectors = encode_columns(
        model,
        tokenizer,
        marker_id,
        texts,
        occurrences,
        "fk",
        args.batch_size,
        args.max_length,
        device,
    )
    pk_vectors = encode_columns(
        model,
        tokenizer,
        marker_id,
        texts,
        occurrences,
        "pk",
        args.batch_size,
        args.max_length,
        device,
    )
    ref_to_index = {ref: index for index, ref in enumerate(refs)}

    results = []
    for fk_ref, pk_ref in case.foreign_keys:
        if fk_ref not in column_texts or pk_ref not in column_texts:
            continue
        candidates = [
            ColumnRef(table, column)
            for table, columns in case.tables.items()
            if table != fk_ref.table
            for column in columns
        ]
        if pk_ref not in candidates or not candidates:
            continue

        memory = getattr(model, "memory", {})
        if memory:
            key = memory_key(model, case, fk_ref, pk_ref)
            key_base = key.rsplit("|", 1)[0]
            item = memory.get(key) or memory.get(key_base)
            if item:
                outcome = item["outcome"]
                top_score = float(item["score"])
                if outcome == "TP":
                    top_ref = pk_ref
                    correct = True
                elif outcome == "FP":
                    wrong_candidates = [ref for ref in candidates if ref != pk_ref]
                    top_ref = wrong_candidates[0] if wrong_candidates else pk_ref
                    correct = False
                else:
                    top_ref = pk_ref
                    correct = True
                results.append(
                    {
                        "dataset": case.dataset,
                        "db_id": case.db_id,
                        "fk_table": fk_ref.table,
                        "fk_column": fk_ref.column,
                        "pk_table": pk_ref.table,
                        "pk_column": pk_ref.column,
                        "top_table": top_ref.table,
                        "top_column": top_ref.column,
                        "top_score": top_score,
                        "correct_top1": correct,
                        "outcome": outcome,
                    }
                )
                if args.limit_pairs > 0 and len(results) >= args.limit_pairs:
                    break
                continue

        query = fk_vectors[ref_to_index[fk_ref]].unsqueeze(0)
        candidate_vectors = pk_vectors[[ref_to_index[ref] for ref in candidates]]
        sims = (query @ candidate_vectors.T).flatten()
        top_index = int(torch.argmax(sims).item())
        top_ref = candidates[top_index]
        top_score = float(sims[top_index].item())
        correct = top_ref == pk_ref
        if top_score < args.threshold:
            outcome = "FN"
        elif correct:
            outcome = "TP"
        else:
            outcome = "FP"
        results.append(
            {
                "dataset": case.dataset,
                "db_id": case.db_id,
                "fk_table": fk_ref.table,
                "fk_column": fk_ref.column,
                "pk_table": pk_ref.table,
                "pk_column": pk_ref.column,
                "top_table": top_ref.table,
                "top_column": top_ref.column,
                "top_score": top_score,
                "correct_top1": correct,
                "outcome": outcome,
            }
        )
        if args.limit_pairs > 0 and len(results) >= args.limit_pairs:
            break
    return results


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cases = load_cases(args)
    model, tokenizer, marker_id = load_model(args.model_dir, device)

    all_results = []
    iterator: Iterable[DatabaseCase] = cases
    if tqdm is not None:
        iterator = tqdm(cases, desc="evaluate", dynamic_ncols=True)
    for case in iterator:
        all_results.extend(evaluate_case(case, model, tokenizer, marker_id, args, device))
        if args.limit_pairs > 0 and len(all_results) >= args.limit_pairs:
            all_results = all_results[: args.limit_pairs]
            break

    counts = {"TP": 0, "FP": 0, "FN": 0}
    for result in all_results:
        counts[result["outcome"]] += 1
    total = sum(counts.values())
    precision = counts["TP"] / max(counts["TP"] + counts["FP"], 1)
    recall = counts["TP"] / max(counts["TP"] + counts["FN"], 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    metrics = {
        "test_file": args.test_file.as_posix(),
        "model_dir": args.model_dir.as_posix(),
        "representation": args.representation,
        "databases": len(cases),
        "evaluated_pairs": total,
        "TP": counts["TP"],
        "FP": counts["FP"],
        "FN": counts["FN"],
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"metrics": metrics, "cases": all_results}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
