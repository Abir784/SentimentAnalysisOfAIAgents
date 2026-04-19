from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.file_management import cleanup_old_files

TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?", re.IGNORECASE)


def _latest_preprocessed(path: str) -> Path:
    p = Path(path) if path else Path("")
    if path and p.exists() and p.is_file():
        return p
    candidates = sorted(Path("data/preprocessed_rule_based").glob("moltbook_preprocessed_rule_based_*.csv"))
    if not candidates:
        raise FileNotFoundError("No preprocessed rule-based CSV found.")
    return candidates[-1]


def _token_count(text: str) -> int:
    return len(TOKEN_RE.findall(str(text or "")))


def _unique_ratio(text: str) -> float:
    toks = TOKEN_RE.findall(str(text or "").lower())
    if not toks:
        return 0.0
    return float(len(set(toks)) / len(toks))


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4: feature extraction for rule-based analysis.")
    parser.add_argument(
        "--input",
        default="",
        help="Path to preprocessed rule-based CSV (defaults to latest).",
    )
    parser.add_argument("--output-dir", default="data/features_rule_based")
    args = parser.parse_args()

    input_path = _latest_preprocessed(args.input)
    df = pd.read_csv(input_path)

    if "text" not in df.columns:
        raise KeyError("Expected column 'text' in preprocessing output.")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feat = pd.DataFrame()
    feat["comment_id"] = df.get("comment_id", pd.Series(range(len(df)))).astype(str)
    feat["post_id"] = df.get("post_id", pd.Series([""] * len(df))).astype(str)
    feat["thread_id"] = df.get("thread_id", pd.Series([""] * len(df))).astype(str)
    feat["author_id"] = df.get("author_id", pd.Series([""] * len(df))).astype(str)

    text_series = df["text"].fillna("").astype(str)
    feat["char_count"] = text_series.str.len().astype(int)
    feat["token_count"] = text_series.map(_token_count).astype(int)
    feat["unique_token_ratio"] = text_series.map(_unique_ratio).astype(float)
    feat["exclamation_count"] = text_series.str.count("!").astype(int)
    feat["question_count"] = text_series.str.count(r"\?").astype(int)
    feat["uppercase_ratio"] = text_series.map(lambda t: (sum(c.isupper() for c in t) / len(t)) if t else 0.0)

    out_csv = out_dir / f"moltbook_features_rule_based_{run_id}.csv"
    summary_path = out_dir / f"moltbook_features_rule_based_summary_{run_id}.json"

    feat.to_csv(out_csv, index=False, encoding="utf-8")

    summary = {
        "run_id": run_id,
        "input_file": str(input_path).replace("\\", "/"),
        "rows": int(len(feat)),
        "feature_columns": [c for c in feat.columns if c not in {"comment_id", "post_id", "thread_id", "author_id"}],
        "output_csv": str(out_csv).replace("\\", "/"),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")

    cleanup_old_files(out_dir, "moltbook_features_rule_based_*.csv", keep_latest=1)
    cleanup_old_files(out_dir, "moltbook_features_rule_based_summary_*.json", keep_latest=1)

    print("Feature extraction complete")
    print(f"input_file: {input_path}")
    print(f"rows: {len(feat)}")
    print(f"output_csv: {out_csv}")
    print(f"summary_path: {summary_path}")


if __name__ == "__main__":
    main()
