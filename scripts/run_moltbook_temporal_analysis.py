from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ruptures as rpt
from statsmodels.stats.proportion import proportion_confint
import statsmodels.formula.api as smf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.file_management import cleanup_old_files


RELATIVE_TIME_RE = re.compile(r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>[smhdw])\s*ago", re.IGNORECASE)
TOKEN_RE = re.compile(r"[A-Za-z']+")


def _parse_relative_time(value: Any) -> Optional[timedelta]:
    text = str(value or "").strip().lower()
    match = RELATIVE_TIME_RE.search(text)
    if match is None:
        return None

    amount = float(match.group("value"))
    unit = match.group("unit").lower()
    if unit == "s":
        return timedelta(seconds=amount)
    if unit == "m":
        return timedelta(minutes=amount)
    if unit == "h":
        return timedelta(hours=amount)
    if unit == "d":
        return timedelta(days=amount)
    if unit == "w":
        return timedelta(weeks=amount)
    return None


def _load_jsonl(path: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def _word_count(text: Any) -> int:
    if not isinstance(text, str):
        return 0
    return len(TOKEN_RE.findall(text))


def _ensure_ensemble_labels(df: pd.DataFrame, input_path: Path) -> pd.DataFrame:
    if "ensemble_label" in df.columns:
        return df

    if input_path.suffix.lower() != ".csv":
        raise ValueError(
            "Temporal analysis requires ensemble_label. Provide a rule-based CSV or run the rule-based scorer first."
        )

    return df


def _derive_actual_timestamp(row: pd.Series) -> Optional[pd.Timestamp]:
    fetched_raw = row.get("fetched_at")
    relative_raw = row.get("relative_time")

    fetched_at = pd.to_datetime(fetched_raw, utc=True, errors="coerce")
    if pd.isna(fetched_at):
        return None

    delta = _parse_relative_time(relative_raw)
    if delta is None:
        return fetched_at

    return fetched_at - delta


def _plot_daily_sentiment(daily: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(daily.index, daily["positive_prop"], label="positive", color="#2a9d8f")
    ax.fill_between(daily.index, daily["positive_lo"], daily["positive_hi"], alpha=0.2, color="#2a9d8f")
    ax.plot(daily.index, daily["neutral_prop"], label="neutral", color="#264653", alpha=0.9)
    ax.plot(daily.index, daily["negative_prop"], label="negative", color="#e76f51", alpha=0.9)
    ax.set_ylabel("Proportion")
    ax.set_title("Daily Sentiment Proportions by Derived Comment Timestamp")
    ax.legend(frameon=False, ncol=3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_change_points(daily: pd.DataFrame, change_points: Iterable[pd.Timestamp], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(daily.index, daily["positive_prop"], label="positive_prop", color="#1d3557")
    for cp in change_points:
        ax.axvline(cp, color="#d62828", linestyle="--", alpha=0.75)
    ax.set_title("Positive Proportion with Detected Change Points")
    ax.set_ylabel("Proportion")
    ax.legend(frameon=False)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run temporal analysis using relative_time + fetched_at to derive actual timestamps.")
    parser.add_argument("--rule-based-input", default="", help="Rule-based CSV with ensemble_label. Defaults to latest CSV in data/rule_based.")
    parser.add_argument("--staged-input", default="data/staged/moltbook_comments_all.jsonl", help="Staged MoltBook JSONL with fetched_at and relative_time.")
    parser.add_argument("--output-dir", default="data/eda", help="Directory for JSON summary and model output.")
    parser.add_argument("--figure-dir", default="data/figures", help="Directory for temporal figures.")
    parser.add_argument("--penalty", type=float, default=1.0, help="PELT penalty for change-point detection.")
    args = parser.parse_args()

    rule_based_input = Path(args.rule_based_input) if args.rule_based_input else Path("")
    if not args.rule_based_input:
        candidates = sorted(Path("data/rule_based").glob("moltbook_rule_based_comments_*.csv"))
        if not candidates:
            raise FileNotFoundError("No rule-based CSV found in data/rule_based")
        rule_based_input = candidates[-1]

    staged_input = Path(args.staged_input)
    if not rule_based_input.exists():
        raise FileNotFoundError(f"Rule-based input not found: {rule_based_input}")
    if not staged_input.exists():
        raise FileNotFoundError(f"Staged input not found: {staged_input}")

    output_dir = Path(args.output_dir)
    figure_dir = Path(args.figure_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    df_rb = pd.read_csv(rule_based_input, dtype={"comment_id": str})
    df_staged = _load_jsonl(staged_input)
    df_staged["comment_id"] = df_staged["comment_id"].astype(str)

    keep_cols = ["comment_id", "relative_time", "fetched_at", "author_id", "post_id", "thread_id", "upvotes", "is_verified"]
    available_cols = [col for col in keep_cols if col in df_staged.columns]
    df = df_rb.merge(df_staged[available_cols], on="comment_id", how="left", suffixes=("", "_staged"))

    if "ensemble_label" not in df.columns:
        raise ValueError("The rule-based input must contain ensemble_label.")

    df["actual_timestamp"] = df.apply(_derive_actual_timestamp, axis=1)
    df = df.dropna(subset=["actual_timestamp"]).copy()
    df["actual_timestamp"] = pd.to_datetime(df["actual_timestamp"], utc=True)
    df["date"] = df["actual_timestamp"].dt.floor("D")
    df["word_count"] = df["text"].fillna("").map(_word_count)
    df["log_upvotes"] = np.log1p(pd.to_numeric(df.get("upvotes", 0), errors="coerce").fillna(0.0))
    df["time_days"] = (df["actual_timestamp"] - df["actual_timestamp"].min()).dt.total_seconds() / 86400.0
    df["is_positive"] = (df["ensemble_label"] == "positive").astype(int)

    daily = df.groupby("date")["ensemble_label"].value_counts().unstack(fill_value=0)
    for label in ["negative", "neutral", "positive"]:
        if label not in daily.columns:
            daily[label] = 0
    daily = daily.sort_index()
    daily["total"] = daily[["negative", "neutral", "positive"]].sum(axis=1)
    for label in ["negative", "neutral", "positive"]:
        daily[f"{label}_prop"] = daily[label] / daily["total"]

    ci = [proportion_confint(int(pos), int(n), method="wilson") if int(n) > 0 else (0.0, 0.0) for pos, n in zip(daily["positive"], daily["total"])]
    daily["positive_lo"] = [lower for lower, upper in ci]
    daily["positive_hi"] = [upper for lower, upper in ci]

    series = daily["positive_prop"].fillna(0.0).to_numpy()
    algo = rpt.Pelt(model="rbf").fit(series)
    breakpoints = algo.predict(pen=float(args.penalty))
    change_points = [daily.index[idx - 1] for idx in breakpoints if 0 < idx < len(daily)]

    model = smf.logit("is_positive ~ time_days + word_count + log_upvotes", data=df)
    if "author_id" in df.columns and df["author_id"].notna().any():
        robust_result = model.fit(
            disp=False,
            cov_type="cluster",
            cov_kwds={"groups": df["author_id"]},
        )
    else:
        robust_result = model.fit(disp=False)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    daily_plot = figure_dir / f"temporal_sentiment_proportions_{run_id}.png"
    change_point_plot = figure_dir / f"temporal_positive_change_points_{run_id}.png"
    model_summary_path = output_dir / f"temporal_model_summary_{run_id}.txt"
    summary_path = output_dir / f"temporal_analysis_summary_{run_id}.json"
    derived_path = output_dir / f"temporal_derived_comments_{run_id}.csv"

    _plot_daily_sentiment(daily, daily_plot)
    _plot_change_points(daily, change_points, change_point_plot)

    df_out = df[[c for c in ["comment_id", "author_id", "post_id", "thread_id", "relative_time", "fetched_at", "actual_timestamp", "ensemble_label", "word_count", "upvotes", "log_upvotes", "time_days", "is_positive"] if c in df.columns]].copy()
    df_out.to_csv(derived_path, index=False, encoding="utf-8")

    model_summary_path.write_text(robust_result.summary().as_text(), encoding="utf-8")

    summary = {
        "rule_based_csv": str(rule_based_input).replace("\\", "/"),
        "staged_jsonl": str(staged_input).replace("\\", "/"),
        "n_rows": int(len(df)),
        "date_range": [str(daily.index.min().date()), str(daily.index.max().date())],
        "change_points": [str(cp.date()) for cp in change_points],
        "model_params": {k: float(v) for k, v in robust_result.params.items()},
        "model_pvalues": {k: float(v) for k, v in robust_result.pvalues.items()},
        "artifacts": {
            "daily_plot": str(daily_plot).replace("\\", "/"),
            "change_point_plot": str(change_point_plot).replace("\\", "/"),
            "model_summary": str(model_summary_path).replace("\\", "/"),
            "derived_comments": str(derived_path).replace("\\", "/"),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    cleanup_old_files(figure_dir, "temporal_sentiment_proportions_*.png", keep_latest=1)
    cleanup_old_files(figure_dir, "temporal_positive_change_points_*.png", keep_latest=1)
    cleanup_old_files(output_dir, "temporal_model_summary_*.txt", keep_latest=1)
    cleanup_old_files(output_dir, "temporal_analysis_summary_*.json", keep_latest=1)
    cleanup_old_files(output_dir, "temporal_derived_comments_*.csv", keep_latest=1)

    print(f"rule_based_input: {rule_based_input}")
    print(f"staged_input: {staged_input}")
    print(f"rows_used: {len(df)}")
    print(f"date_range: {summary['date_range'][0]} -> {summary['date_range'][1]}")
    print(f"change_points: {summary['change_points']}")
    print(f"daily_plot: {daily_plot}")
    print(f"change_point_plot: {change_point_plot}")
    print(f"model_summary: {model_summary_path}")
    print(f"summary_json: {summary_path}")
    print(f"derived_comments: {derived_path}")


if __name__ == "__main__":
    main()