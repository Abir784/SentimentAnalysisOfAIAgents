from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipelines.polarity_moltbook import run_polarity_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stricter MoltBook polarity preprocessing and export modeling artifacts.")
    parser.add_argument(
        "--input",
        default="data/staged/moltbook_comments_all.jsonl",
        help="Input staged comments JSONL file.",
    )
    parser.add_argument(
        "--preprocessed-dir",
        default="data/preprocessed",
        help="Directory for preprocessed JSONL and training-ready CSV outputs.",
    )
    parser.add_argument(
        "--polarity-dir",
        default="data/polarity",
        help="Directory for polarity-scored JSONL and summary JSON outputs.",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Optional run id override, otherwise uses current UTC timestamp.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    result = run_polarity_pipeline(
        input_path=input_path,
        preprocessed_dir=Path(args.preprocessed_dir),
        polarity_dir=Path(args.polarity_dir),
        run_id=args.run_id or None,
    )

    print("Polarity pipeline complete")
    print(f"input_file: {result['input_path']}")
    print(f"raw_rows: {result['raw_rows']}")
    print(f"rows_after_preprocessing: {result['rows_after_preprocessing']}")
    print(f"preprocessed_path: {result['paths']['preprocessed']}")
    print(f"polarity_path: {result['paths']['polarity']}")
    print(f"training_csv_path: {result['paths']['training_csv']}")
    print(f"summary_path: {result['paths']['summary']}")


if __name__ == "__main__":
    main()