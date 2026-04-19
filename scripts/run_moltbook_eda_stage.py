from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipelines.eda_moltbook import build_eda_summary, read_jsonl
from src.utils.file_management import cleanup_old_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3: EDA for rule-based pipeline.")
    parser.add_argument("--input", default="data/staged/moltbook_comments_all.jsonl")
    parser.add_argument("--output-dir", default="data/eda_rule_based")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary_path = output_dir / f"moltbook_eda_rule_based_summary_{run_id}.json"

    rows = read_jsonl(input_path)
    summary = build_eda_summary(rows)
    summary.update(
        {
            "run_id": run_id,
            "input_file": str(input_path).replace("\\", "/"),
            "input_rows": int(len(rows)),
        }
    )

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    cleanup_old_files(output_dir, "moltbook_eda_rule_based_summary_*.json", keep_latest=1)

    print("EDA stage complete")
    print(f"summary_path: {summary_path}")


if __name__ == "__main__":
    main()
