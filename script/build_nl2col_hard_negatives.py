#!/usr/bin/env python3
"""Build nl2column hard negatives from col2col hard-negative pairs.

Each input hard-negative pair contains two FK columns from different tables that
reference the same PK column. For nl2col, we describe one FK column in natural
language and use the other FK column as the hard negative candidate.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterator


VALUE_RE = re.compile(r"<Column-Value>(.*?)(?=<Column>|$)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("dataset/train/hard_negatives"),
        help="Directory containing col2col hard-negative JSONL shards.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/train/nl2col_hard_negatives"),
    )
    parser.add_argument("--shard-size", type=int, default=100_000)
    parser.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="Maximum input records to consume. 0 means consume all records.",
    )
    return parser.parse_args()


def iter_jsonl(input_dir: Path) -> Iterator[dict[str, Any]]:
    for path in sorted(input_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def sample_values(representation: dict[str, str]) -> list[str]:
    text = representation.get("column-only") or next(iter(representation.values()), "")
    match = VALUE_RE.search(text)
    if not match:
        return []
    values = [value.strip() for value in match.group(1).split(",")]
    return [value for value in values if value][:3]


def summary_for(
    database: str,
    table: str,
    column: str,
    representation: dict[str, str],
    shared_pk_table: str,
    shared_pk_column: str,
) -> str:
    values = sample_values(representation)
    pieces = [
        f"The {column} column in the {table} table belongs to the {database} database.",
        f"It is a foreign-key-like column associated with {shared_pk_table}.{shared_pk_column}.",
    ]
    if values:
        pieces.append(f"Example values include {', '.join(values)}.")
    return " ".join(pieces)


def make_record(
    source: dict[str, Any],
    described_side: str,
    candidate_side: str,
) -> dict[str, Any]:
    database = source["database"]
    described_table = source[f"table{described_side}"]
    described_column = source[f"column{described_side}"]
    described_representation = source[f"representation{described_side}"]
    candidate_table = source[f"table{candidate_side}"]
    candidate_column = source[f"column{candidate_side}"]
    candidate_representation = source[f"representation{candidate_side}"]
    shared_pk_table = source.get("shared_pk_table", "")
    shared_pk_column = source.get("shared_pk_column", "")
    return {
        "task": "nl2col_hard_negative",
        "database": database,
        "described_table": described_table,
        "described_column": described_column,
        "summary": summary_for(
            database,
            described_table,
            described_column,
            described_representation,
            shared_pk_table,
            shared_pk_column,
        ),
        "table": candidate_table,
        "column": candidate_column,
        "column_representations": candidate_representation,
        "relation": "hard_negative",
        "shared_pk_table": shared_pk_table,
        "shared_pk_column": shared_pk_column,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    shard_index = 0
    shard_count = 0
    total = 0
    consumed = 0
    handle = None

    def open_shard(index: int):
        path = args.output_dir / f"nl2col_hard_negatives_part{index}.jsonl"
        return path.open("w", encoding="utf-8")

    try:
        handle = open_shard(shard_index)
        for source in iter_jsonl(args.input_dir):
            if args.max_records and consumed >= args.max_records:
                break
            consumed += 1
            for described_side, candidate_side in (("A", "B"), ("B", "A")):
                if shard_count >= args.shard_size:
                    handle.close()
                    shard_index += 1
                    shard_count = 0
                    handle = open_shard(shard_index)
                record = make_record(source, described_side, candidate_side)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                shard_count += 1
                total += 1
    finally:
        if handle is not None:
            handle.close()

    metadata = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "input_records": consumed,
        "output_records": total,
        "shard_size": args.shard_size,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
