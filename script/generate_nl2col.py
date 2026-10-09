#!/usr/bin/env python3
"""Generate nl2col positive training pairs from WikiDBs schemas.

Each JSONL record contains a natural-language semantic summary and the
corresponding column representations. Negatives are intended to be supplied by
in-batch negatives during contrastive training, so this script only stores
positive pairs.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path
from typing import Any


REPRESENTATION_KEYS = (
    "column-only",
    "column+tablename",
    "schema+context",
    "column+context",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-list", type=Path, default=Path("splits/WikiDBs/train.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/train/nl2col"))
    parser.add_argument("--target", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-size", type=int, default=5)
    parser.add_argument("--max-rows-per-table", type=int, default=50)
    parser.add_argument("--records-per-shard", type=int, default=100_000)
    parser.add_argument("--progress-interval", type=int, default=50_000)
    parser.add_argument("--limit-databases", type=int, default=None)
    return parser.parse_args()


def clean_value(value: Any) -> str:
    text = "" if value is None else str(value)
    return " ".join(text.replace("\n", " ").replace("\r", " ").split())


def humanize_identifier(value: str) -> str:
    text = clean_value(value)
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def first_nonempty(values: list[str] | None) -> str:
    if not values:
        return ""
    for value in values:
        cleaned = humanize_identifier(value)
        if cleaned:
            return cleaned
    return ""


def value_text(values: list[str]) -> str:
    return ", ".join(values)


class JsonlShardWriter:
    def __init__(self, output_dir: Path, records_per_shard: int) -> None:
        self.output_dir = output_dir
        self.records_per_shard = records_per_shard
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_index = 0
        self.in_shard = 0
        self.total = 0
        self.handle = None

    def _open_next(self) -> None:
        if self.handle is not None:
            self.handle.close()
        path = self.output_dir / f"nl2col_part{self.shard_index}.jsonl"
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


def table_columns(table: dict[str, Any]) -> list[str]:
    return [column.get("column_name", "") for column in table.get("columns", [])]


def sample_table_values(
    db_path: Path,
    table: dict[str, Any],
    sample_size: int,
    max_rows: int,
) -> dict[str, list[str]]:
    table_name = table.get("table_name", "")
    file_name = table.get("file_name") or f"{table_name}.csv"
    column_names = table_columns(table)
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

    return {
        "column-only": f"<Column>{target_column}<Value>{value_text(target_values)}",
        "column+tablename": (
            f"<Table>{table_name}<Column>{target_column}<Value>{value_text(target_values)}"
        ),
        "schema+context": schema_context,
        "column+context": "".join(context_parts),
    }


def table_description(schema: dict[str, Any], table: dict[str, Any]) -> str:
    table_name = humanize_identifier(table.get("table_name", ""))
    alt_table = first_nonempty(table.get("alternative_table_names"))
    db_name = humanize_identifier(schema.get("database_name", ""))
    topic = humanize_identifier(schema.get("wikidata_topic_item_label", ""))
    prop = humanize_identifier(schema.get("wikidata_property_label", ""))

    table_label = alt_table or table_name
    if topic and prop:
        return f"The {table_label} table belongs to the {db_name} database about {topic} and its {prop} related records."
    if topic:
        return f"The {table_label} table belongs to the {db_name} database about {topic}."
    return f"The {table_label} table belongs to the {db_name} database."


def fk_target_map(table: dict[str, Any]) -> dict[str, tuple[str, str]]:
    targets = {}
    for fk in table.get("foreign_keys", []):
        column = fk.get("column_name", "")
        ref_table = fk.get("reference_table_name", "")
        ref_column = fk.get("reference_column_name", "")
        if column and ref_table and ref_column:
            targets[column] = (ref_table, ref_column)
    return targets


def is_pk_like(table: dict[str, Any], column_name: str) -> bool:
    if not column_name:
        return False
    first_column = table.get("columns", [{}])[0].get("column_name", "")
    name = column_name.lower()
    return column_name == first_column or name in {"id", f"{table.get('table_name', '').lower()}_id"}


def column_description(
    schema: dict[str, Any],
    table: dict[str, Any],
    column: dict[str, Any],
    sampled_values: dict[str, list[str]],
) -> str:
    table_name = humanize_identifier(table.get("table_name", ""))
    table_desc = table_description(schema, table)
    column_name = column.get("column_name", "")
    column_label = first_nonempty(column.get("alternative_column_names")) or humanize_identifier(column_name)
    data_type = humanize_identifier(column.get("data_type", ""))
    property_id = column.get("wikidata_property_id")
    values = sampled_values.get(column_name, [])
    fk_targets = fk_target_map(table)

    pieces = [table_desc]
    pieces.append(f"The {column_label} column in {table_name} stores {data_type or 'values'}")
    if property_id:
        pieces[-1] += f" associated with Wikidata property {property_id}"
    pieces[-1] += "."

    if column_name in fk_targets:
        ref_table, ref_column = fk_targets[column_name]
        pieces.append(
            f"It is a foreign-key-like column that links to {humanize_identifier(ref_table)}.{humanize_identifier(ref_column)}."
        )
    elif is_pk_like(table, column_name):
        pieces.append("It can identify or label records in this table.")

    if values:
        pieces.append(f"Example values include {value_text(values[:3])}.")

    return " ".join(pieces)


def make_record(
    database: str,
    table: dict[str, Any],
    column: dict[str, Any],
    summary: str,
    representations: dict[str, str],
) -> dict[str, Any]:
    table_name = table.get("table_name", "")
    column_name = column.get("column_name", "")
    return {
        "task": "nl2col_positive",
        "database": database,
        "table": table_name,
        "column": column_name,
        "summary": summary,
        "column_representations": representations,
        "relation": "positive",
        "column_type": column.get("data_type", ""),
        "wikidata_property_id": column.get("wikidata_property_id"),
    }


def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    db_paths = read_train_paths(args.train_list, args.seed, args.limit_databases)
    writer = JsonlShardWriter(args.output_dir, args.records_per_shard)
    rng = random.Random(args.seed)

    counts = {"records": 0, "databases": 0, "skipped": 0}
    next_report = args.progress_interval

    try:
        for db_index, db_path in enumerate(db_paths, 1):
            if counts["records"] >= args.target:
                break
            schema = load_schema(db_path)
            if schema is None:
                counts["skipped"] += 1
                continue
            counts["databases"] += 1
            database = schema.get("database_name") or db_path.name
            tables = list(schema.get("tables", []))
            rng.shuffle(tables)

            for table in tables:
                if counts["records"] >= args.target:
                    break
                columns = list(table.get("columns", []))
                if not columns:
                    continue
                rng.shuffle(columns)
                column_order = table_columns(table)
                sampled = sample_table_values(
                    db_path, table, args.sample_size, args.max_rows_per_table
                )
                for column in columns:
                    if counts["records"] >= args.target:
                        break
                    column_name = column.get("column_name", "")
                    if not column_name:
                        continue
                    reps = build_representations(
                        table.get("table_name", ""), column_name, column_order, sampled
                    )
                    summary = column_description(schema, table, column, sampled)
                    writer.write(make_record(database, table, column, summary, reps))
                    counts["records"] += 1

                    if counts["records"] >= next_report:
                        print(
                            f"db={db_index}/{len(db_paths)} records={counts['records']}/{args.target}",
                            flush=True,
                        )
                        while next_report <= counts["records"]:
                            next_report += args.progress_interval
    finally:
        writer.close()

    metadata = {
        "train_list": args.train_list.as_posix(),
        "output_dir": args.output_dir.as_posix(),
        "target": args.target,
        "counts": counts,
        "seed": args.seed,
        "sample_size": args.sample_size,
        "max_rows_per_table": args.max_rows_per_table,
        "records_per_shard": args.records_per_shard,
        "representation_keys": list(REPRESENTATION_KEYS),
        "negative_strategy": "in-batch negatives during training",
    }
    write_metadata(args.output_dir / "metadata.json", metadata)
    print(json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
