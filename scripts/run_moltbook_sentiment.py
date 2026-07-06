from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipelines.eda_moltbook import build_eda_summary, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EDA for MoltBook consolidated comments.")
    parser.add_argument(
        "--input",
        default="data/staged/moltbook_comments_all.jsonl",
        help="Input consolidated comments JSONL file.",
    )
    parser.add_argument(
        "--summary-output",
        default="",
        help="Optional output path for EDA summary JSON.",
    )
    args = parser.parse_args()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    
    eda_dir = Path("data/eda")
    eda_dir.mkdir(parents=True, exist_ok=True)

    summary_path = (
        Path(args.summary_output)
        if args.summary_output
        else eda_dir / f"moltbook_eda_summary_{run_id}.json"
    )

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    rows = read_jsonl(input_path)

    summary = build_eda_summary(rows)
    summary["run_id"] = run_id
    summary["input_file"] = str(input_path)
    summary["input_rows"] = len(rows)
    summary["output_path"] = str(summary_path)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")

    print("EDA complete")
    print(f"input_file: {input_path}")
    print(f"input_rows: {len(rows)}")
    print(f"summary_path: {summary_path}")
    print(f"row_count: {summary['row_count']}")
    print(f"unique_posts: {summary['unique_posts']}")
    print(f"duplicate_rows: {summary['duplicate_rows_by_platform_post_comment']}")


if __name__ == "__main__":
    main()
