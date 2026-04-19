from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipelines.polarity_moltbook import preprocess_for_sentiment, traditional_preprocess
from src.utils.file_management import cleanup_old_files


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2: text preprocessing for rule-based pipeline.")
    parser.add_argument("--input", default="data/staged/moltbook_comments_all.jsonl")
    parser.add_argument("--output-dir", default="data/preprocessed_rule_based")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    rows = _read_jsonl(input_path)
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No rows found in staged input.")

    if "text" not in df.columns:
        raise KeyError("Missing required column: text")

    df["text"] = df["text"].fillna("").astype(str)
    df["text_basic_clean"] = df["text"].map(preprocess_for_sentiment)
    df["text_traditional_clean"] = df["text"].map(traditional_preprocess)
    df["text_len_words_basic_clean"] = df["text_basic_clean"].str.split().str.len().fillna(0).astype(int)
    df["text_len_words_traditional_clean"] = (
        df["text_traditional_clean"].str.split().str.len().fillna(0).astype(int)
    )

    keep_cols = [
        c
        for c in [
            "comment_id",
            "post_id",
            "thread_id",
            "parent_id",
            "author_id",
            "is_verified",
            "upvotes",
            "text",
            "text_basic_clean",
            "text_traditional_clean",
            "text_len_words_basic_clean",
            "text_len_words_traditional_clean",
        ]
        if c in df.columns
    ]
    out_df = df[keep_cols].copy()

    csv_path = output_dir / f"moltbook_preprocessed_rule_based_{run_id}.csv"
    jsonl_path = output_dir / f"moltbook_preprocessed_rule_based_{run_id}.jsonl"
    summary_path = output_dir / f"moltbook_preprocessed_rule_based_summary_{run_id}.json"

    out_df.to_csv(csv_path, index=False, encoding="utf-8")
    _write_jsonl(jsonl_path, out_df.to_dict(orient="records"))

    summary = {
        "run_id": run_id,
        "input_file": str(input_path).replace("\\", "/"),
        "rows": int(len(out_df)),
        "output_csv": str(csv_path).replace("\\", "/"),
        "output_jsonl": str(jsonl_path).replace("\\", "/"),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")

    cleanup_old_files(output_dir, "moltbook_preprocessed_rule_based_*.csv", keep_latest=1)
    cleanup_old_files(output_dir, "moltbook_preprocessed_rule_based_*.jsonl", keep_latest=1)
    cleanup_old_files(output_dir, "moltbook_preprocessed_rule_based_summary_*.json", keep_latest=1)

    print("Text preprocessing complete")
    print(f"rows: {len(out_df)}")
    print(f"output_csv: {csv_path}")
    print(f"output_jsonl: {jsonl_path}")
    print(f"summary_path: {summary_path}")


if __name__ == "__main__":
    main()
