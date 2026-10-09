#!/usr/bin/env python3
"""Build three FK-PK discovery training datasets from WikiDBs.

Outputs JSONL shards for:
  (a) FK-PK positive pairs from schema foreign keys.
  (b) table-structure invariance positives from row/column swaps.
  (c) hard negatives: two FK columns pointing to the same PK.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import random
from pathlib import Path
from typing import Any, Iterable


REPRESENTATION_KEYS = (
    "column-only",
    "column+tablename",
    "schema+context",
    "column+context",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-list", type=Path, default=Path("splits/WikiDBs/train.txt"))
    parser.add_argument("--output-root", type=Path, default=Path("dataset/train"))
    parser.add_argument("--target-a", type=int, default=1_500_000)
    parser.add_argument("--target-b", type=int, default=300_000)
    parser.add_argument("--target-c", type=int, default=150_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-size", type=int, default=5)
    parser.add_argument("--max-rows-per-table", type=int, default=50)
    parser.add_argument("--records-per-shard", type=int, default=100_000)
    parser.add_argument("--progress-interval", type=int, default=10_000)
    parser.add_argument("--limit-databases", type=int, default=None)
    return parser.parse_args()


def stable_seed(seed: int, *parts: str) -> int:
    payload = "\0".join([str(seed), *parts]).encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def clean_value(value: Any) -> str:
    text = "" if value is None else str(value)
    return " ".join(text.replace("\n", " ").replace("\r", " ").split())


def value_text(values: list[str]) -> str:
    return ", ".join(values)


class JsonlShardWriter:
    def __init__(self, output_dir: Path, shard_prefix: str, records_per_shard: int) -> None:
        self.output_dir = output_dir
        self.shard_prefix = shard_prefix
        self.records_per_shard = records_per_shard
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_index = 0
        self.total = 0
        self.in_shard = 0
        self.handle = None

    def _open_next(self) -> None:
        if self.handle is not None:
            self.handle.close()
        path = self.output_dir / f"{self.shard_prefix}_part{self.shard_index}.jsonl"
        self.handle = path.open("w", encoding="utf-8")
        self.shard_index += 1
        self.in_shard = 0

    def write(self, record: dict[str, Any]) -> None:
        if self.handle is None or self.in_shard >= self.records_per_shard:
            self._open_next()
        assert self.handle is not None
        self.handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.total += 1
        self.in_shard += 1

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None


def read_train_paths(train_list: Path, seed: int, limit: int | None) -> list[Path]:
    paths = [Path(row) for row in train_list.read_text(encoding="utf-8").splitlines() if row]
    rng = random.Random(seed)
    rng.shuffle(paths)
    if limit is not None:
        paths = paths[:limit]
    return paths


def load_schema(db_path: Path) -> dict[str, Any] | None:
    try:
        return json.loads((db_path / "schema.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def table_columns(schema: dict[str, Any]) -> dict[str, list[str]]:
    return {
        table.get("table_name", ""): [
            column.get("column_name", "") for column in table.get("columns", [])
        ]
        for table in schema.get("tables", [])
    }


def table_files(schema: dict[str, Any]) -> dict[str, str]:
    return {
        table.get("table_name", ""): table.get("file_name") or f"{table.get('table_name', '')}.csv"
        for table in schema.get("tables", [])
    }


def sample_table_values(
    db_path: Path,
    table_name: str,
    file_name: str,
    column_names: list[str],
    sample_size: int,
    max_rows: int,
) -> dict[str, list[str]]:
    values = {column: [] for column in column_names}
    seen = {column: set() for column in column_names}
    csv_path = db_path / "tables" / file_name
    if not csv_path.exists():
        return values

    try:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return values
            available = [column for column in column_names if column in reader.fieldnames]
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


def build_representations(
    table_name: str,
    target_column: str,
    column_order: list[str],
    sampled_values: dict[str, list[str]],
) -> dict[str, str]:
    target_values = sampled_values.get(target_column, [])
    schema_context = "".join(
        [f"<Table>{table_name}"]
        + [f"<Column{index}>{name}" for index, name in enumerate(column_order, 1)]
    )

    context_parts = [f"<Table>{table_name}"]
    for index, name in enumerate(column_order, 1):
        context_parts.append(f"<Column{index}>{name}")
        context_parts.append(f"<Column{index}-Value>{value_text(sampled_values.get(name, []))}")
    column_context = "".join(context_parts)

    return {
        "column-only": f"<Column>{target_column}<Value>{value_text(target_values)}",
        "column+tablename": (
            f"<Table>{table_name}<Column>{target_column}<Value>{value_text(target_values)}"
        ),
        "schema+context": schema_context,
        "column+context": column_context,
    }


def build_database_rep_cache(
    db_path: Path,
    schema: dict[str, Any],
    sample_size: int,
    max_rows: int,
) -> dict[tuple[str, str], dict[str, str]]:
    columns_by_table = table_columns(schema)
    files_by_table = table_files(schema)
    cache = {}
    for table_name, column_names in columns_by_table.items():
        sampled = sample_table_values(
            db_path, table_name, files_by_table.get(table_name, ""), column_names, sample_size, max_rows
        )
        for column_name in column_names:
            cache[(table_name, column_name)] = build_representations(
                table_name, column_name, column_names, sampled
            )
    return cache


def build_rep_cache_for_columns(
    db_path: Path,
    schema: dict[str, Any],
    needed_columns: set[tuple[str, str]],
    sample_size: int,
    max_rows: int,
) -> dict[tuple[str, str], dict[str, str]]:
    columns_by_table = table_columns(schema)
    files_by_table = table_files(schema)
    needed_by_table: dict[str, set[str]] = {}
    for table_name, column_name in needed_columns:
        needed_by_table.setdefault(table_name, set()).add(column_name)

    cache = {}
    for table_name, needed_names in needed_by_table.items():
        column_names = columns_by_table.get(table_name, [])
        if not column_names:
            continue
        sampled = sample_table_values(
            db_path, table_name, files_by_table.get(table_name, ""), column_names, sample_size, max_rows
        )
        for column_name in needed_names:
            if column_name in column_names:
                cache[(table_name, column_name)] = build_representations(
                    table_name, column_name, column_names, sampled
                )
    return cache


def fk_pk_pairs(schema: dict[str, Any]) -> list[tuple[str, str, str, str]]:
    cols = {
        (table.get("table_name", ""), column.get("column_name", ""))
        for table in schema.get("tables", [])
        for column in table.get("columns", [])
    }
    pairs = []
    for table in schema.get("tables", []):
        fk_table = table.get("table_name", "")
        for fk in table.get("foreign_keys", []):
            fk_col = fk.get("column_name", "")
            pk_table = fk.get("reference_table_name", "")
            pk_col = fk.get("reference_column_name", "")
            if (fk_table, fk_col) in cols and (pk_table, pk_col) in cols:
                pairs.append((fk_table, fk_col, pk_table, pk_col))
    return pairs


def make_pair_record(
    task: str,
    database: str,
    table_a: str,
    column_a: str,
    reps_a: dict[str, str],
    table_b: str,
    column_b: str,
    reps_b: dict[str, str],
    relation: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
        "task": task,
        "database": database,
        "tableA": table_a,
        "columnA": column_a,
        "representationA": reps_a,
        "tableB": table_b,
        "columnB": column_b,
        "representationB": reps_b,
        "relation": relation,
    }
    if extra:
        record.update(extra)
    return record


def shuffled_pair_order(items: Iterable[Any], rng: random.Random) -> list[Any]:
    rows = list(items)
    rng.shuffle(rows)
    return rows


def swap_two_positions(values: list[str], rng: random.Random) -> list[str]:
    swapped = list(values)
    if len(swapped) >= 2:
        i, j = rng.sample(range(len(swapped)), 2)
        swapped[i], swapped[j] = swapped[j], swapped[i]
    return swapped


def build_augmented_reps(
    db_path: Path,
    schema: dict[str, Any],
    table_name: str,
    column_name: str,
    sample_size: int,
    max_rows: int,
    rng: random.Random,
) -> tuple[dict[str, str], dict[str, str], str]:
    columns_by_table = table_columns(schema)
    files_by_table = table_files(schema)
    column_names = columns_by_table[table_name]
    sampled = sample_table_values(
        db_path, table_name, files_by_table.get(table_name, ""), column_names, sample_size, max_rows
    )
    reps_a = build_representations(table_name, column_name, column_names, sampled)

    augmentation = "column_swap" if len(column_names) >= 2 and rng.random() < 0.5 else "row_swap"
    if augmentation == "column_swap":
        order_b = list(column_names)
        i, j = rng.sample(range(len(order_b)), 2)
        order_b[i], order_b[j] = order_b[j], order_b[i]
        values_b = sampled
    else:
        order_b = column_names
        values_b = {name: swap_two_positions(vals, rng) for name, vals in sampled.items()}

    reps_b = build_representations(table_name, column_name, order_b, values_b)
    return reps_a, reps_b, augmentation


def build_augmented_reps_from_sampled(
    table_name: str,
    column_name: str,
    column_names: list[str],
    sampled: dict[str, list[str]],
    rng: random.Random,
) -> tuple[dict[str, str], dict[str, str], str]:
    reps_a = build_representations(table_name, column_name, column_names, sampled)

    augmentation = "column_swap" if len(column_names) >= 2 and rng.random() < 0.5 else "row_swap"
    if augmentation == "column_swap":
        order_b = list(column_names)
        i, j = rng.sample(range(len(order_b)), 2)
        order_b[i], order_b[j] = order_b[j], order_b[i]
        values_b = sampled
    else:
        order_b = column_names
        values_b = {name: swap_two_positions(vals, rng) for name, vals in sampled.items()}

    reps_b = build_representations(table_name, column_name, order_b, values_b)
    return reps_a, reps_b, augmentation


def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def maybe_report(
    counts: dict[str, int],
    targets: dict[str, int],
    db_index: int,
    total_dbs: int,
    next_report: int,
    interval: int,
) -> int:
    total_written = counts["a"] + counts["b"] + counts["c"]
    if total_written >= next_report:
        print(
            f"db={db_index}/{total_dbs} a={counts['a']}/{targets['a']} "
            f"b={counts['b']}/{targets['b']} c={counts['c']}/{targets['c']}",
            flush=True,
        )
        while next_report <= total_written:
            next_report += interval
    return next_report


def main() -> None:
    args = parse_args()
    db_paths = read_train_paths(args.train_list, args.seed, args.limit_databases)
    rng = random.Random(args.seed)

    out_a = JsonlShardWriter(args.output_root / "wikidbs", "wikidbs", args.records_per_shard)
    out_b = JsonlShardWriter(args.output_root / "synthesis", "synthesis", args.records_per_shard)
    out_c = JsonlShardWriter(
        args.output_root / "hard_negatives", "hard_negatives", args.records_per_shard
    )

    counts = {"a": 0, "b": 0, "c": 0, "databases": 0, "skipped": 0}
    targets = {"a": args.target_a, "b": args.target_b, "c": args.target_c}
    next_report = args.progress_interval

    try:
        for db_index, db_path in enumerate(db_paths, 1):
            if counts["a"] >= args.target_a and counts["b"] >= args.target_b and counts["c"] >= args.target_c:
                break

            schema = load_schema(db_path)
            if schema is None:
                counts["skipped"] += 1
                continue
            counts["databases"] += 1
            database_name = schema.get("database_name") or db_path.name
            pairs = shuffled_pair_order(fk_pk_pairs(schema), rng)

            need_rep_cache = counts["a"] < args.target_a or counts["c"] < args.target_c
            needed_columns: set[tuple[str, str]] = set()
            if need_rep_cache:
                for fk_table, fk_col, pk_table, pk_col in pairs:
                    needed_columns.add((fk_table, fk_col))
                    needed_columns.add((pk_table, pk_col))
            rep_cache = (
                build_rep_cache_for_columns(
                    db_path,
                    schema,
                    needed_columns,
                    args.sample_size,
                    args.max_rows_per_table,
                )
                if need_rep_cache
                else {}
            )

            if counts["a"] < args.target_a:
                for fk_table, fk_col, pk_table, pk_col in pairs:
                    out_a.write(
                        make_pair_record(
                            "fk_pk_positive",
                            database_name,
                            fk_table,
                            fk_col,
                            rep_cache[(fk_table, fk_col)],
                            pk_table,
                            pk_col,
                            rep_cache[(pk_table, pk_col)],
                            "positive",
                        )
                    )
                    counts["a"] += 1
                    next_report = maybe_report(
                        counts, targets, db_index, len(db_paths), next_report, args.progress_interval
                    )
                    if counts["a"] >= args.target_a:
                        break

            if counts["c"] < args.target_c:
                grouped: dict[tuple[str, str], list[tuple[str, str]]] = {}
                for fk_table, fk_col, pk_table, pk_col in pairs:
                    grouped.setdefault((pk_table, pk_col), []).append((fk_table, fk_col))
                pk_groups = shuffled_pair_order(grouped.items(), rng)
                for (pk_table, pk_col), fks in pk_groups:
                    if len(fks) < 2:
                        continue
                    rng.shuffle(fks)
                    for (left_table, left_col), (right_table, right_col) in itertools.combinations(fks, 2):
                        out_c.write(
                            make_pair_record(
                                "same_pk_fk_hard_negative",
                                database_name,
                                left_table,
                                left_col,
                                rep_cache[(left_table, left_col)],
                                right_table,
                                right_col,
                                rep_cache[(right_table, right_col)],
                                "hard_negative",
                                {"shared_pk_table": pk_table, "shared_pk_column": pk_col},
                            )
                        )
                        counts["c"] += 1
                        next_report = maybe_report(
                            counts, targets, db_index, len(db_paths), next_report, args.progress_interval
                        )
                        if counts["c"] >= args.target_c:
                            break
                    if counts["c"] >= args.target_c:
                        break

            if counts["b"] < args.target_b:
                columns_by_table = table_columns(schema)
                files_by_table = table_files(schema)
                table_names = shuffled_pair_order(columns_by_table.keys(), rng)
                for table_name in table_names:
                    column_names = columns_by_table[table_name]
                    if not column_names:
                        continue
                    sampled = sample_table_values(
                        db_path,
                        table_name,
                        files_by_table.get(table_name, ""),
                        column_names,
                        args.sample_size,
                        args.max_rows_per_table,
                    )
                    shuffled_columns = shuffled_pair_order(column_names, rng)
                    for column_name in shuffled_columns:
                        reps_a, reps_b, augmentation = build_augmented_reps_from_sampled(
                            table_name, column_name, column_names, sampled, rng
                        )
                        out_b.write(
                            make_pair_record(
                                "table_structure_positive",
                                database_name,
                                table_name,
                                column_name,
                                reps_a,
                                table_name,
                                column_name,
                                reps_b,
                                "positive",
                                {"augmentation": augmentation},
                            )
                        )
                        counts["b"] += 1
                        next_report = maybe_report(
                            counts, targets, db_index, len(db_paths), next_report, args.progress_interval
                        )
                        if counts["b"] >= args.target_b:
                            break
                    if counts["b"] >= args.target_b:
                        break

    finally:
        out_a.close()
        out_b.close()
        out_c.close()

    metadata = {
        "train_list": args.train_list.as_posix(),
        "output_root": args.output_root.as_posix(),
        "targets": {"a": args.target_a, "b": args.target_b, "c": args.target_c},
        "counts": counts,
        "seed": args.seed,
        "sample_size": args.sample_size,
        "max_rows_per_table": args.max_rows_per_table,
        "records_per_shard": args.records_per_shard,
        "representation_keys": list(REPRESENTATION_KEYS),
    }
    write_metadata(args.output_root / "metadata.json", metadata)
    print(json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
