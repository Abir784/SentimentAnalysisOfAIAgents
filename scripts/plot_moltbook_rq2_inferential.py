from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABEL_ORDER = ["negative", "neutral", "positive"]
METHOD_ORDER = ["vader", "swn", "ensemble"]
METHOD_DISPLAY = {
    "vader": "VADER",
    "swn": "SentiWordNet",
    "ensemble": "Ensemble",
}
METHOD_COLOR = {
    "vader": "#1f77b4",
    "swn": "#ff7f0e",
    "ensemble": "#2ca02c",
}


def _extract_run_id(path: Path) -> str:
    stem = path.stem
    parts = stem.split("_")
    if parts and parts[-1].endswith("Z"):
        return parts[-1]
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _find_latest_stats_table(input_path: str) -> Path:
    if input_path:
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        return path

    candidates = sorted(Path("data/rule_based").glob("moltbook_rq2_stats_table_*.csv"))
    if not candidates:
        raise FileNotFoundError("No RQ2 stats table found under data/rule_based")
    return candidates[-1]


def _row(df: pd.DataFrame, metric_type: str, method: str, label: str = "") -> pd.Series:
    filtered = df[(df["metric_type"] == metric_type) & (df["method"] == method)]
    if label:
        filtered = filtered[filtered["label"] == label]
    if filtered.empty:
        raise ValueError(f"Missing row for metric_type={metric_type}, method={method}, label={label}")
    return filtered.iloc[0]


def _to_float(v) -> float:
    return float(v)


def _build_figure(df: pd.DataFrame, run_id: str, output_path: Path) -> None:
    fig = plt.figure(figsize=(14, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[2.0, 1.4])

    ax_share = fig.add_subplot(gs[0, :])
    ax_mean = fig.add_subplot(gs[1, 0])
    ax_pair = fig.add_subplot(gs[1, 1])

    # Panel A: label shares with 95% CI
    x = np.arange(len(LABEL_ORDER), dtype=float)
    width = 0.25

    for i, method in enumerate(METHOD_ORDER):
        points: List[float] = []
        err_low: List[float] = []
        err_high: List[float] = []
        for label in LABEL_ORDER:
            r = _row(df, "label_share", method, label)
            p = _to_float(r["point"])
            lo = _to_float(r["ci95_low"])
            hi = _to_float(r["ci95_high"])
            points.append(p)
            err_low.append(p - lo)
            err_high.append(hi - p)

        centers = x + (i - 1) * width
        ax_share.bar(
            centers,
            points,
            width=width,
            color=METHOD_COLOR[method],
            alpha=0.85,
            label=METHOD_DISPLAY[method],
            zorder=2,
        )
        ax_share.errorbar(
            centers,
            points,
            yerr=np.vstack([err_low, err_high]),
            fmt="none",
            ecolor="black",
            elinewidth=1,
            capsize=3,
            zorder=3,
        )

    ax_share.set_xticks(x)
    ax_share.set_xticklabels([lbl.capitalize() for lbl in LABEL_ORDER])
    ax_share.set_ylabel("Label share")
    ax_share.set_ylim(0, 0.8)
    ax_share.grid(axis="y", linestyle="--", alpha=0.3, zorder=1)
    ax_share.legend(frameon=False, ncol=3, loc="upper left")
    ax_share.set_title("RQ2 Panel A: Sentiment label shares with 95% bootstrap CI")

    # Panel B: mean scores with 95% CI
    mean_metrics = [
        ("vader_compound", "VADER compound", "#1f77b4"),
        ("swn_score", "SentiWordNet score", "#ff7f0e"),
    ]
    mean_x = np.arange(len(mean_metrics), dtype=float)
    mean_points = []
    mean_err_low = []
    mean_err_high = []
    for metric_key, _, _ in mean_metrics:
        r = _row(df, "mean_score", metric_key)
        p = _to_float(r["point"])
        lo = _to_float(r["ci95_low"])
        hi = _to_float(r["ci95_high"])
        mean_points.append(p)
        mean_err_low.append(p - lo)
        mean_err_high.append(hi - p)

    colors = [c for _, _, c in mean_metrics]
    labels = [lbl for _, lbl, _ in mean_metrics]
    ax_mean.bar(mean_x, mean_points, color=colors, alpha=0.85, zorder=2)
    ax_mean.errorbar(
        mean_x,
        mean_points,
        yerr=np.vstack([mean_err_low, mean_err_high]),
        fmt="none",
        ecolor="black",
        elinewidth=1,
        capsize=3,
        zorder=3,
    )
    ax_mean.set_xticks(mean_x)
    ax_mean.set_xticklabels(labels, rotation=12, ha="right")
    ax_mean.set_ylabel("Mean score")
    ax_mean.grid(axis="y", linestyle="--", alpha=0.3, zorder=1)
    ax_mean.set_title("RQ2 Panel B: Mean score with 95% bootstrap CI")

    # Panel C: paired positive-rate difference + McNemar p-value
    r_diff = _row(df, "paired_rate_difference", "vader_positive_minus_swn_positive")
    r_p = _row(df, "mcnemar_exact_p", "vader_positive_vs_swn_positive")
    diff_point = _to_float(r_diff["point"])
    diff_lo = _to_float(r_diff["ci95_low"])
    diff_hi = _to_float(r_diff["ci95_high"])
    p_exact = _to_float(r_p["point"])

    ax_pair.axhline(0.0, color="gray", linewidth=1, linestyle="--", zorder=1)
    ax_pair.errorbar(
        [0],
        [diff_point],
        yerr=[[diff_point - diff_lo], [diff_hi - diff_point]],
        fmt="o",
        color="#8c564b",
        markersize=7,
        capsize=5,
        zorder=3,
    )
    ax_pair.set_xlim(-0.8, 0.8)
    ax_pair.set_xticks([0])
    ax_pair.set_xticklabels(["VADER - SWN positive rate"])
    ax_pair.set_ylabel("Rate difference")
    ax_pair.set_title("RQ2 Panel C: Paired difference and McNemar test")
    ax_pair.grid(axis="y", linestyle="--", alpha=0.3, zorder=1)
    ax_pair.text(
        0.02,
        0.95,
        f"Exact p = {p_exact:.3e}\n95% CI = [{diff_lo:.4f}, {diff_hi:.4f}]",
        transform=ax_pair.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )

    fig.suptitle(f"MoltBook RQ2 Inferential Summary (run {run_id})", fontsize=14, fontweight="bold")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _write_metadata(output_path: Path, run_id: str, input_table: Path) -> None:
    payload: Dict[str, str] = {
        "run_id": run_id,
        "input_table": input_table.as_posix(),
        "output_figure": output_path.as_posix(),
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    meta_path = output_path.with_suffix(".json")
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a publication-ready RQ2 inferential figure from the RQ2 stats table."
    )
    parser.add_argument(
        "--input",
        default="",
        help="Path to RQ2 stats table CSV. Defaults to latest data/rule_based/moltbook_rq2_stats_table_*.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="data/rule_based",
        help="Directory for generated figure output.",
    )
    args = parser.parse_args()

    input_table = _find_latest_stats_table(args.input)
    run_id = _extract_run_id(input_table)

    df = pd.read_csv(input_table)
    required_columns = ["metric_type", "method", "label", "point", "ci95_low", "ci95_high"]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in stats table: {missing}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"moltbook_rq2_inferential_summary_{run_id}.png"

    _build_figure(df, run_id, output_path)
    _write_metadata(output_path, run_id, input_table)

    print(f"Generated inferential figure: {output_path}")


if __name__ == "__main__":
    main()
