from __future__ import annotations

import argparse
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import importlib.metadata as metadata
import numpy as np
import pandas as pd

LABELS = ["negative", "neutral", "positive"]


def _extract_run_id(path: Path) -> str:
    stem = path.stem
    parts = stem.split("_")
    if parts and parts[-1].endswith("Z"):
        return parts[-1]
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _bootstrap_ci(values: np.ndarray, func, n_bootstrap: int, seed: int) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = values.shape[0]
    estimates = np.empty(n_bootstrap, dtype=float)

    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        estimates[i] = float(func(values[idx]))

    point = float(func(values))
    lo = float(np.percentile(estimates, 2.5))
    hi = float(np.percentile(estimates, 97.5))
    return point, lo, hi


def _bootstrap_label_share(
    labels: np.ndarray,
    target: str,
    n_bootstrap: int,
    seed: int,
) -> Tuple[float, float, float]:
    numeric = (labels == target).astype(float)
    return _bootstrap_ci(numeric, np.mean, n_bootstrap, seed)


def _mcnemar_from_binary(a: np.ndarray, b: np.ndarray) -> Dict[str, float | int]:
    a = a.astype(bool)
    b = b.astype(bool)

    n00 = int((~a & ~b).sum())
    n01 = int((~a & b).sum())
    n10 = int((a & ~b).sum())
    n11 = int((a & b).sum())

    discordant = n01 + n10
    if discordant == 0:
        return {
            "n00": n00,
            "n01": n01,
            "n10": n10,
            "n11": n11,
            "chi2_cc": 0.0,
            "p_value_chi2_cc": 1.0,
            "p_value_exact_binomial": 1.0,
        }

    chi2_cc = ((abs(n01 - n10) - 1.0) ** 2) / discordant
    p_value_chi2_cc = math.erfc(math.sqrt(chi2_cc / 2.0))

    k = min(n01, n10)
    two_sided = 2.0 * sum(math.comb(discordant, i) for i in range(0, k + 1)) / (2.0**discordant)
    p_value_exact_binomial = min(1.0, float(two_sided))

    return {
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "n11": n11,
        "chi2_cc": float(chi2_cc),
        "p_value_chi2_cc": float(p_value_chi2_cc),
        "p_value_exact_binomial": p_value_exact_binomial,
    }


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
    output_files: List[Path],
    seed: int,
    n_bootstrap: int,
    output_json: Path,
    output_csv: Path,
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
        "experiment": "rq2_stats",
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "input_file": input_file.as_posix(),
        "output_files": [p.as_posix() for p in output_files],
        "parameters": {
            "seed": seed,
            "n_bootstrap": n_bootstrap,
            "paired_test": "mcnemar_positive_detection",
        },
        "environment": {
            "python": platform.python_version(),
            "direct_requirements": req,
            "installed_direct_packages": installed,
        },
    }

    out_manifest = manifest_dir / f"rq2_stats_manifest_{run_id}.json"
    out_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute RQ2 bootstrap CIs and paired significance tests for rule-based sentiment outputs."
    )
    parser.add_argument(
        "--input",
        default="",
        help="Path to rule-based comments CSV. Defaults to latest data/rule_based/moltbook_rule_based_comments_*.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="data/rule_based",
        help="Directory for output summary files.",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=10000,
        help="Number of bootstrap resamples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    input_path = _find_latest_comments_csv(args.input)
    run_id = _extract_run_id(input_path)

    df = pd.read_csv(input_path)
    required = ["vader_label", "swn_label", "ensemble_label", "vader_compound", "swn_score"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_share_ci: Dict[str, Dict[str, Dict[str, float]]] = {}
    for method_col in ["vader_label", "swn_label", "ensemble_label"]:
        method_name = method_col.replace("_label", "")
        labels = df[method_col].astype(str).to_numpy()
        label_share_ci[method_name] = {}

        for i, label in enumerate(LABELS):
            point, lo, hi = _bootstrap_label_share(labels, label, args.bootstrap, args.seed + i)
            label_share_ci[method_name][label] = {
                "point": round(point, 6),
                "ci95_low": round(lo, 6),
                "ci95_high": round(hi, 6),
            }

    mean_ci = {}
    for i, score_col in enumerate(["vader_compound", "swn_score"]):
        values = df[score_col].astype(float).to_numpy()
        point, lo, hi = _bootstrap_ci(values, np.mean, args.bootstrap, args.seed + 100 + i)
        mean_ci[score_col] = {
            "point": round(point, 6),
            "ci95_low": round(lo, 6),
            "ci95_high": round(hi, 6),
        }

    vader_positive = (df["vader_label"].astype(str) == "positive").to_numpy()
    swn_positive = (df["swn_label"].astype(str) == "positive").to_numpy()
    mcnemar_positive = _mcnemar_from_binary(vader_positive, swn_positive)

    positive_rate_diff = vader_positive.astype(float) - swn_positive.astype(float)
    diff_point, diff_lo, diff_hi = _bootstrap_ci(
        positive_rate_diff,
        np.mean,
        args.bootstrap,
        args.seed + 200,
    )

    summary = {
        "run_id": run_id,
        "input_file": str(input_path).replace("\\", "/"),
        "rows": int(len(df)),
        "seed": args.seed,
        "n_bootstrap": args.bootstrap,
        "rq2_outputs": {
            "label_share_bootstrap_ci95": label_share_ci,
            "mean_score_bootstrap_ci95": mean_ci,
            "paired_significance_positive_detection": {
                "comparison": "vader_positive_vs_swn_positive",
                "mcnemar": mcnemar_positive,
                "paired_rate_difference_bootstrap_ci95": {
                    "point": round(float(diff_point), 6),
                    "ci95_low": round(float(diff_lo), 6),
                    "ci95_high": round(float(diff_hi), 6),
                },
            },
        },
    }

    out_json = output_dir / f"moltbook_rq2_stats_{run_id}.json"
    with out_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    rows: List[Dict[str, object]] = []
    for method_name, method_block in label_share_ci.items():
        for label, payload in method_block.items():
            rows.append(
                {
                    "metric_type": "label_share",
                    "method": method_name,
                    "label": label,
                    "point": payload["point"],
                    "ci95_low": payload["ci95_low"],
                    "ci95_high": payload["ci95_high"],
                }
            )

    for score_name, payload in mean_ci.items():
        rows.append(
            {
                "metric_type": "mean_score",
                "method": score_name,
                "label": "",
                "point": payload["point"],
                "ci95_low": payload["ci95_low"],
                "ci95_high": payload["ci95_high"],
            }
        )

    rows.append(
        {
            "metric_type": "paired_rate_difference",
            "method": "vader_positive_minus_swn_positive",
            "label": "",
            "point": round(float(diff_point), 6),
            "ci95_low": round(float(diff_lo), 6),
            "ci95_high": round(float(diff_hi), 6),
        }
    )

    rows.append(
        {
            "metric_type": "mcnemar_exact_p",
            "method": "vader_positive_vs_swn_positive",
            "label": "",
            "point": summary["rq2_outputs"]["paired_significance_positive_detection"]["mcnemar"]["p_value_exact_binomial"],
            "ci95_low": "",
            "ci95_high": "",
        }
    )

    out_csv = output_dir / f"moltbook_rq2_stats_table_{run_id}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    _write_manifest(
        run_id=run_id,
        input_file=input_path,
        output_files=[out_json, out_csv],
        seed=args.seed,
        n_bootstrap=args.bootstrap,
        output_json=out_json,
        output_csv=out_csv,
    )


if __name__ == "__main__":
    main()
