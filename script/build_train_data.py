#!/usr/bin/env python3
"""Build demo/train data from a WikiDBs raw-data directory.

This is a small driver around the data builders used in the main project. It
creates:
  - wikidbs FK-PK positive pairs
  - synthesis structure-preserving positives
  - col2col hard negatives
  - nl2col positives
  - nl2col hard negatives
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wikidbs-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("demo/dataset/train_full"))
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--limit-databases", type=int, default=0)
    parser.add_argument("--target-wikidbs", type=int, default=1_500_000)
    parser.add_argument("--target-synthesis", type=int, default=300_000)
    parser.add_argument("--target-hard", type=int, default=150_000)
    parser.add_argument("--target-nl2col", type=int, default=500_000)
    parser.add_argument("--target-nl2col-hard", type=int, default=300_000)
    parser.add_argument("--records-per-shard", type=int, default=100_000)
    parser.add_argument("--sample-size", type=int, default=5)
    parser.add_argument("--max-rows-per-table", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def discover_databases(root: Path, limit: int) -> list[Path]:
    paths = sorted(path.parent for path in root.glob("part-*/*/schema.json"))
    if limit > 0:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"No WikiDBs databases found under {root}")
    return paths


def run(cmd: list[str]) -> None:
    print("==>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    args.output_root.mkdir(parents=True, exist_ok=True)

    split_file = args.split_file or args.output_root / "wikidbs_train_paths.txt"
    if args.split_file is None:
        db_paths = discover_databases(args.wikidbs_root, args.limit_databases)
        split_file.write_text(
            "\n".join(path.as_posix() for path in db_paths) + "\n",
            encoding="utf-8",
        )

    common = [
        "--train-list",
        split_file.as_posix(),
        "--seed",
        str(args.seed),
        "--sample-size",
        str(args.sample_size),
        "--max-rows-per-table",
        str(args.max_rows_per_table),
        "--records-per-shard",
        str(args.records_per_shard),
    ]
    if args.limit_databases > 0:
        common.extend(["--limit-databases", str(args.limit_databases)])

    run(
        [
            sys.executable,
            (script_dir / "build_wikidbs_three_part_train.py").as_posix(),
            "--output-root",
            args.output_root.as_posix(),
            "--target-a",
            str(args.target_wikidbs),
            "--target-b",
            str(args.target_synthesis),
            "--target-c",
            str(args.target_hard),
            *common,
        ]
    )

    run(
        [
            sys.executable,
            (script_dir / "generate_nl2col.py").as_posix(),
            "--output-dir",
            (args.output_root / "nl2col").as_posix(),
            "--target",
            str(args.target_nl2col),
            *common,
        ]
    )

    run(
        [
            sys.executable,
            (script_dir / "build_nl2col_hard_negatives.py").as_posix(),
            "--input-dir",
            (args.output_root / "hard_negatives").as_posix(),
            "--output-dir",
            (args.output_root / "nl2col_hard_negatives").as_posix(),
            "--shard-size",
            str(args.records_per_shard),
            "--max-records",
            str(args.target_nl2col_hard),
        ]
    )


if __name__ == "__main__":
    main()
