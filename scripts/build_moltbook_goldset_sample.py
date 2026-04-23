from __future__ import annotations

import argparse
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import importlib.metadata as metadata
import pandas as pd


def _extract_run_id(path: Path) -> str:
    stem = path.stem
    parts = stem.split("_")
    if parts and parts[-1].endswith("Z"):
        return parts[-1]
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _find_latest_comments_csv(input_path: str) -> Path:
    if input_path:
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        return path

    candidates = sorted(Path("data/rule_based").glob("moltbook_rule_based_comments_*.csv"))
    if not candidates:
        raise FileNotFoundError("No rule-based comments CSV found under data/rule_based")
    return candidates[-1]


def _allocate_counts(group_sizes: Dict[str, int], target_n: int) -> Dict[str, int]:
    total = sum(group_sizes.values())
    if total == 0:
        return {k: 0 for k in group_sizes}

    raw = {k: (v / total) * target_n for k, v in group_sizes.items()}
    alloc = {k: min(group_sizes[k], int(math.floor(raw[k]))) for k in group_sizes}

    remaining = target_n - sum(alloc.values())
    if remaining <= 0:
        return alloc

    remainders = sorted(
        ((raw[k] - alloc[k], k) for k in group_sizes),
        reverse=True,
    )

    idx = 0
    while remaining > 0 and idx < len(remainders):
        _, key = remainders[idx]
        if alloc[key] < group_sizes[key]:
            alloc[key] += 1
            remaining -= 1
        else:
            idx += 1

    if remaining > 0:
        for key in group_sizes:
            while remaining > 0 and alloc[key] < group_sizes[key]:
                alloc[key] += 1
                remaining -= 1

    return alloc


def _read_direct_requirements(path: Path) -> Dict[str, str]:
    pinned: Dict[str, str] = {}
    if not path.exists():
        return pinned

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        pkg, version = line.split("==", 1)
        pinned[pkg.strip()] = version.strip()
    return pinned


def _write_manifest(
    run_id: str,
    input_file: Path,
    sample_size: int,
    target_size: int,
    seed: int,
    sample_csv: Path,
    strata_csv: Path,
    summary_txt: Path,
) -> None:
    manifest_dir = Path("data/manifests")
    manifest_dir.mkdir(parents=True, exist_ok=True)

    req = _read_direct_requirements(Path("requirements.txt"))
    installed: Dict[str, str] = {}
    for pkg in req:
        try:
            installed[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            installed[pkg] = "not-installed"

    manifest = {
        "experiment": "goldset_sample",
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "input_file": input_file.as_posix(),
        "output_files": [sample_csv.as_posix(), strata_csv.as_posix(), summary_txt.as_posix()],
        "parameters": {
            "seed": seed,
            "target_size": target_size,
            "sample_size": sample_size,
            "stratification": "ensemble_label_x_is_verified",
        },
        "environment": {
            "python": platform.python_version(),
            "direct_requirements": req,
            "installed_direct_packages": installed,
        },
    }

    out_manifest = manifest_dir / f"goldset_sample_manifest_{run_id}.json"
    out_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a stratified sentiment gold-set annotation sample (300-500 rows)."
    )
    parser.add_argument(
        "--input",
        default="",
        help="Path to rule-based comments CSV. Defaults to latest data/rule_based/moltbook_rule_based_comments_*.csv",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=400,
        help="Target number of rows in the annotation sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/gold",
        help="Output directory for gold-set artifacts.",
    )
    args = parser.parse_args()

    if args.target_size < 300 or args.target_size > 500:
        raise ValueError("target-size must be between 300 and 500 inclusive")

    input_path = _find_latest_comments_csv(args.input)
    run_id = _extract_run_id(input_path)

    df = pd.read_csv(input_path)
    required_cols = [
        "comment_id",
        "post_id",
        "thread_id",
        "author_id",
        "text",
        "is_verified",
        "upvotes",
        "vader_label",
        "swn_label",
        "ensemble_label",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work = df.copy()
    work["is_verified"] = work["is_verified"].astype(str)
    work["stratum"] = work["ensemble_label"].astype(str) + "|" + work["is_verified"].astype(str)

    group_sizes = work.groupby("stratum").size().to_dict()
    allocations = _allocate_counts(group_sizes, min(args.target_size, len(work)))

    samples: List[pd.DataFrame] = []
    for stratum, n_take in allocations.items():
        if n_take <= 0:
            continue
        stratum_df = work[work["stratum"] == stratum]
        samples.append(stratum_df.sample(n=n_take, random_state=args.seed))

    sampled = pd.concat(samples, ignore_index=True)
    sampled = sampled.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    sampled = sampled[
        [
            "comment_id",
            "post_id",
            "thread_id",
            "author_id",
            "is_verified",
            "upvotes",
            "text",
            "vader_label",
            "swn_label",
            "ensemble_label",
        ]
    ].copy()

    sampled["rater_1_label"] = ""
    sampled["rater_2_label"] = ""
    sampled["adjudicated_label"] = ""
    sampled["adjudication_notes"] = ""

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_csv = output_dir / f"moltbook_goldset_sample_{run_id}.csv"
    sampled.to_csv(sample_csv, index=False)

    strata_summary = (
        sampled.groupby(["ensemble_label", "is_verified"]).size().reset_index(name="sample_count")
    )
    strata_csv = output_dir / f"moltbook_goldset_sample_strata_{run_id}.csv"
    strata_summary.to_csv(strata_csv, index=False)

    summary_txt = output_dir / f"moltbook_goldset_sample_summary_{run_id}.txt"
    summary_txt.write_text(
        "\n".join(
            [
                f"run_id: {run_id}",
                f"input_file: {input_path.as_posix()}",
                f"target_size: {args.target_size}",
                f"sample_size: {len(sampled)}",
                f"seed: {args.seed}",
                f"sample_csv: {sample_csv.as_posix()}",
                f"strata_csv: {strata_csv.as_posix()}",
            ]
        ),
        encoding="utf-8",
    )

    _write_manifest(
        run_id=run_id,
        input_file=input_path,
        sample_size=len(sampled),
        target_size=args.target_size,
        seed=args.seed,
        sample_csv=sample_csv,
        strata_csv=strata_csv,
        summary_txt=summary_txt,
    )


if __name__ == "__main__":
    main()
