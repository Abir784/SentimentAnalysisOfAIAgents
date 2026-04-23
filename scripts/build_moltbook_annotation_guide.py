from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

LABELS = ["negative", "neutral", "positive"]


def _find_latest(pattern: str) -> Path:
    files = sorted(Path().glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    return files[-1]


def _extract_run_id(path: Path) -> str:
    parts = path.stem.split("_")
    if parts and parts[-1].endswith("Z"):
        return parts[-1]
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_manifest(run_id: str, params: Dict[str, Any], outputs: List[Path]) -> Path:
    manifest_dir = Path("data/manifests")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    out = manifest_dir / f"annotation_batch_manifest_{run_id}.json"
    out.write_text(
        json.dumps(
            {
                "experiment": "gold_annotation_batch",
                "run_id": run_id,
                "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "parameters": params,
                "output_files": [p.as_posix() for p in outputs],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build stratified annotation batch when gold labels are unavailable.")
    parser.add_argument("--input", default="", help="Rule-based comments CSV; defaults to latest")
    parser.add_argument("--output", default="data/gold/annotation_batch_01.csv", help="Output annotation CSV")
    parser.add_argument("--sample-size", type=int, default=50, help="Total sample size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--run-id", default="", help="Optional run ID")
    args = parser.parse_args()

    in_path = Path(args.input) if args.input else _find_latest("data/rule_based/moltbook_rule_based_comments_*.csv")
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    run_id = args.run_id.strip() or _extract_run_id(in_path)
    rng = np.random.default_rng(args.seed)

    df = pd.read_csv(in_path)
    required = ["comment_id", "text", "ensemble_label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    take_per_class = args.sample_size // 3
    remainder = args.sample_size - (take_per_class * 3)
    per_class = {"negative": take_per_class, "neutral": take_per_class, "positive": take_per_class}
    for label in LABELS[:remainder]:
        per_class[label] += 1

    sampled_parts: List[pd.DataFrame] = []
    for label in LABELS:
        sub = df[df["ensemble_label"].astype(str) == label].copy()
        n_take = min(per_class[label], len(sub))
        if n_take <= 0:
            continue
        idx = rng.choice(sub.index.to_numpy(), size=n_take, replace=False)
        sampled_parts.append(sub.loc[idx])

    sampled = pd.concat(sampled_parts, ignore_index=True)
    sampled = sampled.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    out_df = sampled[["comment_id", "text", "ensemble_label"]].copy()
    out_df = out_df.rename(columns={"ensemble_label": "model_label"})
    out_df["human_label"] = ""
    out_df.to_csv(out_path, index=False)

    guide_text = """ANNOTATION GUIDE (3-class sentiment)

Classes:
- negative: clear negative judgment, criticism, frustration, harm, or rejection.
- neutral: informational, descriptive, procedural, or mixed/unclear affect with no dominant polarity.
- positive: clear approval, praise, support, optimism, or favorable evaluation.

Rules:
1) Label the dominant sentiment of the full comment, not isolated words.
2) If strong positive and negative cues are balanced or ambiguous, choose neutral.
3) Ignore model label when assigning human_label.
4) If sarcasm/irony cannot be resolved confidently, choose neutral.
5) Keep labels lowercase: negative/neutral/positive.
"""
    guide_path = Path("data/gold/annotation_batch_01_guide.txt")
    guide_path.write_text(guide_text, encoding="utf-8")

    manifest_path = _write_manifest(
        run_id,
        {
            "seed": args.seed,
            "input": in_path.as_posix(),
            "sample_size": int(len(out_df)),
            "target_sample_size": int(args.sample_size),
            "stratification": "ensemble_label",
        },
        [out_path, guide_path],
    )

    print("Annotation batch generated")
    print(f"output_csv: {out_path}")
    print(f"rows: {len(out_df)}")
    print(guide_text)
    print(f"manifest_path: {manifest_path}")


if __name__ == "__main__":
    main()
