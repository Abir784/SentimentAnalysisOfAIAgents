"""
RQ3: Sentiment Dynamics Within Conversation Threads

Research Question:
How does sentiment evolve within conversation threads, and is there evidence of
sentiment contagion, polarization, or balancing across reply sequences?

Analysis includes:
1. Parent-child sentiment alignment (do replies match parent sentiment?)
2. Sentiment trajectory by thread depth (is sentiment stable or changing?)
3. Author-level consistency (do individual authors maintain consistent sentiment?)
4. Evidence of sentiment contagion vs. polarization
5. Sentiment diversity within threads
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, spearmanr, wilcoxon

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.figsize": (12, 8), "font.size": 11, "figure.dpi": 150})

LABELS = ["negative", "neutral", "positive"]
LABEL_TO_INT = {"negative": -1, "neutral": 0, "positive": 1}


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


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _prepare_df(rb_path: Path, staged_path: Path) -> pd.DataFrame:
    """
    Merge rule-based sentiment labels with staged metadata including thread structure.
    """
    rb = pd.read_csv(rb_path)
    staged = pd.read_json(staged_path, lines=True, orient="records")

    if isinstance(staged, dict):
        staged = pd.DataFrame([staged])
    elif not isinstance(staged, pd.DataFrame):
        staged = pd.DataFrame(staged)

    use_cols_rb = [c for c in ["comment_id", "ensemble_label", "vader_compound"] if c in rb.columns]
    use_cols_staged = [c for c in ["comment_id", "thread_id", "post_id", "author_id", "level", "text", "upvotes", "is_verified"] if c in staged.columns]

    merged = rb[use_cols_rb].merge(staged[use_cols_staged], on="comment_id", how="inner")

    merged["thread_depth"] = pd.to_numeric(merged.get("level", 0), errors="coerce").fillna(0)
    if merged["thread_depth"].nunique(dropna=True) <= 1:
        merged["thread_depth"] = merged.groupby("thread_id").cumcount().astype(float)

    merged["text_len_words"] = merged.get("text", "").fillna("").astype(str).str.split().str.len()
    merged["is_verified"] = merged.get("is_verified", False).astype(str).str.lower().isin(["1", "true", "yes"])
    merged["sentiment_int"] = merged["ensemble_label"].map(LABEL_TO_INT)

    return merged


def _build_thread_reply_structure(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Build parent-child relationships within each thread to analyze reply patterns.
    Returns a dict mapping thread_id -> list of (parent_label, child_label) pairs.
    """
    structure = {}

    for thread_id in df["thread_id"].unique():
        thread_df = df[df["thread_id"] == thread_id].sort_values("thread_depth").reset_index(drop=True)

        if len(thread_df) < 2:
            continue

        pairs = []
        for idx, row in thread_df.iterrows():
            if idx > 0:
                parent_label = thread_df.loc[idx - 1, "ensemble_label"]
                child_label = row["ensemble_label"]
                pairs.append((parent_label, child_label))

        structure[thread_id] = pairs

    return structure


def _compute_sentiment_alignment(structure: Dict[str, Any]) -> pd.DataFrame:
    """
    Compute parent-child sentiment alignment statistics.
    Returns DataFrame with alignment rates and contingency table info.
    """
    all_pairs = []
    for thread_pairs in structure.values():
        all_pairs.extend(thread_pairs)

    if not all_pairs:
        return pd.DataFrame()

    parent_labels = [p[0] for p in all_pairs]
    child_labels = [p[1] for p in all_pairs]

    contingency = pd.crosstab(parent_labels, child_labels, margins=True)

    exact_match = sum(1 for p, c in all_pairs if p == c)
    alignment_rate = exact_match / len(all_pairs)

    chi2, p_val, dof, expected = chi2_contingency(pd.crosstab(parent_labels, child_labels))

    alignment_df = pd.DataFrame(
        [
            {
                "metric": "exact_match_rate",
                "value": alignment_rate,
                "total_pairs": len(all_pairs),
            },
            {
                "metric": "chi2_statistic",
                "value": chi2,
                "p_value": p_val,
            },
        ]
    )

    return alignment_df, contingency


def _compute_sentiment_trajectory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze sentiment distribution by thread depth (position in thread).
    """
    trajectories = []

    for thread_id in df["thread_id"].unique():
        thread_df = df[df["thread_id"] == thread_id].sort_values("thread_depth")

        if len(thread_df) < 2:
            continue

        thread_depth_groups = thread_df.groupby(pd.cut(thread_df["thread_depth"], bins=3, labels=["early", "mid", "late"]))

        for phase, phase_df in thread_depth_groups:
            for label in LABELS:
                count = (phase_df["ensemble_label"] == label).sum()
                prop = count / len(phase_df) if len(phase_df) > 0 else 0
                trajectories.append(
                    {
                        "thread_id": thread_id,
                        "phase": phase,
                        "sentiment": label,
                        "count": count,
                        "proportion": prop,
                    }
                )

    trajectory_df = pd.DataFrame(trajectories)
    return trajectory_df


def _compute_author_sentiment_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Measure per-author sentiment consistency: do authors repeatedly express
    the same sentiment across multiple replies, or is their sentiment variable?
    """
    consistency = []

    for author_id in df["author_id"].unique():
        author_df = df[df["author_id"] == author_id]

        if len(author_df) < 2:
            continue

        sentiments = author_df["ensemble_label"].value_counts()
        n_comments = len(author_df)

        most_frequent_label = sentiments.index[0]
        most_frequent_count = sentiments.iloc[0]
        consistency_score = most_frequent_count / n_comments

        sentiment_diversity = len(sentiments) / len(LABELS)

        consistency.append(
            {
                "author_id": author_id,
                "n_comments": n_comments,
                "dominant_sentiment": most_frequent_label,
                "dominant_proportion": consistency_score,
                "sentiment_diversity": sentiment_diversity,
                "n_unique_sentiments": len(sentiments),
            }
        )

    return pd.DataFrame(consistency)


def _plot_alignment_heatmap(contingency: pd.DataFrame, out_path: Path) -> None:
    """
    Visualize parent-child sentiment contingency as heatmap.
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    contingency_numeric = contingency.iloc[:-1, :-1]
    normalized = contingency_numeric.div(contingency_numeric.sum(axis=1), axis=0)

    sns.heatmap(normalized, annot=True, fmt=".2%", cmap="YlGnBu", ax=ax, cbar_kws={"label": "Proportion"})
    ax.set_title("Parent → Child Sentiment Transition Probabilities")
    ax.set_xlabel("Child Comment Sentiment")
    ax.set_ylabel("Parent Comment Sentiment")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_trajectory_by_phase(trajectory_df: pd.DataFrame, out_path: Path) -> None:
    """
    Visualize sentiment composition across thread phases (early, mid, late).
    """
    if trajectory_df.empty:
        return

    phase_order = ["early", "mid", "late"]
    color_map = {"negative": "#e76f51", "neutral": "#9e9e9e", "positive": "#2a9d8f"}

    agg = trajectory_df.groupby(["phase", "sentiment"])["proportion"].mean().reset_index()
    agg["phase"] = pd.Categorical(agg["phase"], categories=phase_order, ordered=True)
    agg = agg.sort_values("phase")

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    for sentiment in LABELS:
        data = agg[agg["sentiment"] == sentiment]
        ax.plot(data["phase"], data["proportion"], marker="o", label=sentiment, color=color_map[sentiment], linewidth=2)

    ax.set_xlabel("Thread Phase")
    ax.set_ylabel("Mean Sentiment Proportion")
    ax.set_title("Sentiment Evolution Within Threads")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_author_consistency_scatter(consistency_df: pd.DataFrame, out_path: Path) -> None:
    """
    Scatter plot of author consistency vs. comment frequency.
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    colors = {"negative": "#e76f51", "neutral": "#9e9e9e", "positive": "#2a9d8f"}
    for sentiment in LABELS:
        mask = consistency_df["dominant_sentiment"] == sentiment
        ax.scatter(
            consistency_df[mask]["n_comments"],
            consistency_df[mask]["dominant_proportion"],
            label=sentiment,
            alpha=0.6,
            s=60,
            color=colors[sentiment],
        )

    ax.set_xlabel("Number of Comments by Author")
    ax.set_ylabel("Consistency Score (% of Dominant Sentiment)")
    ax.set_title("Author Sentiment Consistency")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="RQ3: Sentiment Dynamics Within Threads")
    parser.add_argument(
        "--rule-based",
        type=str,
        default="",
        help="Path to rule-based CSV (auto-detect if empty)",
    )
    parser.add_argument(
        "--staged",
        type=str,
        default="",
        help="Path to staged JSONL (auto-detect if empty)",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default="data/figures",
        help="Output directory for figures",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="",
        help="Run ID for output files",
    )
    args = parser.parse_args()

    rb_path = Path(args.rule_based.strip()) if args.rule_based.strip() else _find_latest("data/rule_based/moltbook_rule_based_comments_*.csv")
    staged_path = Path(args.staged.strip()) if args.staged.strip() else _find_latest("data/staged/moltbook_comments_all.jsonl")

    for p in [rb_path, staged_path]:
        if not p.exists():
            raise FileNotFoundError(f"Input not found: {p}")

    run_id = args.run_id.strip() or _extract_run_id(rb_path)
    out_dir = Path(args.figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _prepare_df(rb_path, staged_path)
    df = df[df["ensemble_label"].isin(LABELS)].copy()

    print(f"Analyzing {len(df)} comments across {df['thread_id'].nunique()} threads")

    structure = _build_thread_reply_structure(df)
    print(f"Built reply structure for {len(structure)} threads with parent-child pairs")

    alignment_df, contingency = _compute_sentiment_alignment(structure)
    alignment_csv = out_dir / f"rq3_alignment_stats_{run_id}.csv"
    alignment_df.to_csv(alignment_csv, index=False)
    contingency_csv = out_dir / f"rq3_contingency_table_{run_id}.csv"
    contingency.to_csv(contingency_csv)

    trajectory_df = _compute_sentiment_trajectory(df)
    trajectory_csv = out_dir / f"rq3_trajectory_by_phase_{run_id}.csv"
    trajectory_df.to_csv(trajectory_csv, index=False)

    consistency_df = _compute_author_sentiment_consistency(df)
    consistency_csv = out_dir / f"rq3_author_consistency_{run_id}.csv"
    consistency_df.to_csv(consistency_csv, index=False)

    alignment_png = out_dir / f"rq3_alignment_heatmap_{run_id}.png"
    _plot_alignment_heatmap(contingency, alignment_png)

    trajectory_png = out_dir / f"rq3_sentiment_trajectory_{run_id}.png"
    _plot_trajectory_by_phase(trajectory_df, trajectory_png)

    consistency_png = out_dir / f"rq3_author_consistency_scatter_{run_id}.png"
    _plot_author_consistency_scatter(consistency_df, consistency_png)

    summary = {
        "run_id": run_id,
        "n_comments": len(df),
        "n_threads": df["thread_id"].nunique(),
        "n_authors": df["author_id"].nunique(),
        "n_posts": df["post_id"].nunique(),
        "alignment_rate": float(alignment_df[alignment_df["metric"] == "exact_match_rate"]["value"].iloc[0]) if len(alignment_df) > 0 else None,
        "mean_author_consistency": float(consistency_df["dominant_proportion"].mean()) if len(consistency_df) > 0 else None,
        "mean_author_sentiment_diversity": float(consistency_df["sentiment_diversity"].mean()) if len(consistency_df) > 0 else None,
    }

    summary_json = out_dir / f"rq3_dynamics_summary_{run_id}.json"
    with summary_json.open("w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"RQ3 sentiment dynamics analysis complete")
    print(f"run_id: {run_id}")
    print(f"alignment_rate: {summary['alignment_rate']:.4f}" if summary['alignment_rate'] is not None else "alignment_rate: N/A")
    print(f"mean_author_consistency: {summary['mean_author_consistency']:.4f}" if summary['mean_author_consistency'] is not None else "mean_author_consistency: N/A")
    print(f"outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
