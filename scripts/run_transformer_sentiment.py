from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipelines.transformer_sentiment import (
    MODEL_SPECS,
    load_moltbook_records,
    load_synthetic_records,
    run_models,
)


DEFAULT_MODELS = ["twitter-roberta", "xlm-twitter", "bertweet"]


def missing_transformer_dependencies() -> list[str]:
    packages = {
        "torch": "torch",
        "transformers": "transformers",
        "sentencepiece": "sentencepiece",
    }
    return [
        package_name
        for import_name, package_name in packages.items()
        if importlib.util.find_spec(import_name) is None
    ]


def add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_SPECS),
        default=DEFAULT_MODELS,
        help="Model aliases to run. Defaults to the three 3-class social-text models.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Inference device. 'auto' prefers CUDA, then MPS, then CPU.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--long-text-strategy",
        choices=["truncate", "window"],
        default="truncate",
        help="Truncate long texts or average probabilities across overlapping windows.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=128,
        help="Token overlap for --long-text-strategy window.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of text records, useful for a smoke test.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run pretrained transformer sentiment models on one dataset at a time."
    )
    subparsers = parser.add_subparsers(dest="dataset", required=True)

    moltbook = subparsers.add_parser("moltbook", help="Score real Moltbook comments.")
    moltbook.add_argument(
        "--input",
        type=Path,
        default=Path("data/staged/moltbook_comments_all.jsonl"),
    )
    moltbook.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/transformer_sentiment/moltbook"),
    )
    add_shared_arguments(moltbook)

    synthetic = subparsers.add_parser("synthetic", help="Score generated conversation messages.")
    synthetic.add_argument(
        "--input",
        type=Path,
        default=Path("data/sythetic/conversations.jsonl"),
        help="The repository currently uses the directory name 'sythetic'.",
    )
    synthetic.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/transformer_sentiment/synthetic"),
    )
    add_shared_arguments(synthetic)
    return parser


def main() -> None:
    parser = build_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        print("\nChoose one dataset explicitly:")
        print("  python scripts/run_transformer_sentiment.py moltbook")
        print("  python scripts/run_transformer_sentiment.py synthetic")
        return

    args = parser.parse_args()
    if not args.input.exists():
        parser.error(f"input file not found: {args.input}")

    missing = missing_transformer_dependencies()
    if missing:
        parser.error(
            "missing transformer dependencies: "
            f"{', '.join(missing)}. Install them with: "
            "python -m pip install -r requirements_transformers.txt"
        )

    if args.dataset == "moltbook":
        records = load_moltbook_records(args.input, limit=args.limit)
    else:
        records = load_synthetic_records(args.input, limit=args.limit)

    print(f"Dataset: {args.dataset}")
    print(f"Input: {args.input}")
    print(f"Text records: {len(records):,}")
    summaries = run_models(
        records=records,
        model_aliases=args.models,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        long_text_strategy=args.long_text_strategy,
        stride=args.stride,
    )

    combined_summary = args.output_dir / "all_models_summary.json"
    combined_summary.write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Combined summary: {combined_summary}")


if __name__ == "__main__":
    main()
