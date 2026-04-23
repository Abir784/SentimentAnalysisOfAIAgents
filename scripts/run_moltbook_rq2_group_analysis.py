from __future__ import annotations

import argparse
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import entropy, kruskal, mannwhitneyu

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.figsize": (10, 6), "font.size": 12, "figure.dpi": 150})

LABELS = ["negative", "neutral", "positive"]
STACK_COLORS = ["#e76f51", "#9e9e9e", "#2a9d8f"]


def _find_latest(pattern: str) -> Path:
    files = sorted(Path().glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {pattern}")
    return files[-1]


def _extract_run_id(path: Path) -> str:
    stem = path.stem
    parts = stem.split("_")
    if parts and parts[-1].endswith("Z"):
        return parts[-1]
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _group_stats(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    dist = (
        df.groupby(group_col)["ensemble_label"]
        .value_counts(normalize=True)
        .unstack(fill_value=0.0)
        .reindex(columns=LABELS, fill_value=0.0)
    )
    dist.columns = [f"prop_{c}" for c in dist.columns]

    meta = df.groupby(group_col).agg(n_comments=("comment_id", "count"), n_authors=("author_id", "nunique"))
    out = dist.join(meta)
    out["dominant_class"] = out[["prop_negative", "prop_neutral", "prop_positive"]].idxmax(axis=1).str.replace("prop_", "", regex=False)
    out["entropy"] = out[["prop_negative", "prop_neutral", "prop_positive"]].apply(
        lambda r: float(entropy(r.to_numpy() + 1e-10)), axis=1
    )
    return out.reset_index()


def _kruskal_binary_by_group(df: pd.DataFrame, group_col: str) -> Tuple[float, float, int]:
    groups = []
    for _, sub in df.groupby(group_col):
        arr = sub["pos_flag"].to_numpy(dtype=float)
        if arr.size >= 2:
            groups.append(arr)
    if len(groups) < 2:
        return 0.0, 1.0, 0
    h, p = kruskal(*groups)
    return float(h), float(p), int(len(groups))


def _pairwise_mwu(df: pd.DataFrame, group_col: str, min_n: int = 10, max_groups: int = 30) -> pd.DataFrame:
    group_sizes = df[group_col].value_counts()
    keep = group_sizes[group_sizes >= min_n].head(max_groups).index.tolist()
    sub = df[df[group_col].isin(keep)].copy()

    rows: List[Dict[str, Any]] = []
    groups = {g: d["pos_flag"].to_numpy(dtype=float) for g, d in sub.groupby(group_col)}
    keys = sorted(groups)

    if len(keys) < 2:
        return pd.DataFrame(columns=[group_col, "group_b", "u_stat", "p_value", "p_bonferroni", "significant"])

    n_tests = len(keys) * (len(keys) - 1) // 2
    for a, b in itertools.combinations(keys, 2):
        u, p = mannwhitneyu(groups[a], groups[b], alternative="two-sided")
        p_adj = min(1.0, p * n_tests)
        rows.append(
            {
                group_col: a,
                "group_b": b,
                "u_stat": float(u),
                "p_value": float(p),
                "p_bonferroni": float(p_adj),
                "significant": bool(p_adj < 0.05),
            }
        )
    return pd.DataFrame(rows)


def _plot_by_post(stats_post: pd.DataFrame, out_png: Path) -> None:
    plot_df = stats_post.sort_values("prop_positive", ascending=True).reset_index(drop=True)
    x = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    bottom = np.zeros(len(plot_df))
    for idx, label in enumerate(LABELS):
        vals = plot_df[f"prop_{label}"].to_numpy()
        ax.bar(x, vals, bottom=bottom, label=label, color=STACK_COLORS[idx], width=0.85)
        bottom += vals

    ax.set_title("RQ2 Sentiment Composition by Post")
    ax.set_xlabel("Post rank (sorted by positive share)")
    ax.set_ylabel("Proportion")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _plot_author_entropy(stats_author: pd.DataFrame, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    vals = stats_author["entropy"].astype(float)
    ax.hist(vals, bins=30, color="#6d597a", edgecolor="white")
    ax.axvline(vals.mean(), linestyle="--", color="#d62828", label=f"mean={vals.mean():.3f}")
    ax.set_title("RQ2 Author-Level Sentiment Entropy")
    ax.set_xlabel("Entropy")
    ax.set_ylabel("Authors")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _plot_top_authors(stats_author: pd.DataFrame, out_png: Path) -> None:
    top = stats_author.sort_values("n_comments", ascending=False).head(20).copy()
    color_map = {"negative": "#e76f51", "neutral": "#9e9e9e", "positive": "#2a9d8f"}

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    for dom, sub in top.groupby("dominant_class"):
        ax.scatter(
            sub["prop_positive"],
            sub["n_comments"],
            s=70,
            color=color_map.get(dom, "#264653"),
            label=dom,
            alpha=0.9,
        )

    ax.set_title("RQ2 Top Authors: Activity vs Positive Share")
    ax.set_xlabel("Positive proportion")
    ax.set_ylabel("Number of comments")
    ax.legend(title="Dominant class")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _write_manifest(run_id: str, params: Dict[str, Any], outputs: List[Path]) -> Path:
    manifest_dir = Path("data/manifests")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_dir / f"rq2_group_analysis_manifest_{run_id}.json"
    path.write_text(
        json.dumps(
            {
                "experiment": "rq2_group_analysis",
                "run_id": run_id,
                "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "parameters": params,
                "output_files": [p.as_posix() for p in outputs],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="RQ2 group-level sentiment analysis by post/thread/author.")
    parser.add_argument("--input", default="", help="Rule-based comments CSV; defaults to latest")
    parser.add_argument("--figures-dir", default="data/figures", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--run-id", default="", help="Optional run ID")
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else _find_latest("data/rule_based/moltbook_rule_based_comments_*.csv")
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    run_id = args.run_id.strip() or _extract_run_id(input_path)
    np.random.seed(args.seed)

    out_dir = Path(args.figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    required = ["comment_id", "post_id", "thread_id", "author_id", "ensemble_label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["ensemble_label"] = df["ensemble_label"].astype(str)
    df["pos_flag"] = (df["ensemble_label"] == "positive").astype(int)

    post_stats = _group_stats(df, "post_id")
    thread_stats = _group_stats(df, "thread_id")
    author_stats = _group_stats(df, "author_id")

    post_csv = out_dir / f"rq2_group_post_stats_{run_id}.csv"
    thread_csv = out_dir / f"rq2_group_thread_stats_{run_id}.csv"
    author_csv = out_dir / f"rq2_group_author_stats_{run_id}.csv"

    post_stats.to_csv(post_csv, index=False)
    thread_stats.to_csv(thread_csv, index=False)
    author_stats.to_csv(author_csv, index=False)

    post_h, post_p, post_k = _kruskal_binary_by_group(df, "post_id")
    thread_h, thread_p, thread_k = _kruskal_binary_by_group(df, "thread_id")
    author_h, author_p, author_k = _kruskal_binary_by_group(df, "author_id")

    post_pairwise = _pairwise_mwu(df, "post_id", min_n=10, max_groups=55)
    author_pairwise = _pairwise_mwu(df, "author_id", min_n=10, max_groups=30)

    post_pairwise_csv = out_dir / f"rq2_pairwise_post_mwu_{run_id}.csv"
    author_pairwise_csv = out_dir / f"rq2_pairwise_author_mwu_{run_id}.csv"
    post_pairwise.to_csv(post_pairwise_csv, index=False)
    author_pairwise.to_csv(author_pairwise_csv, index=False)

    by_post_png = out_dir / f"rq2_by_post_{run_id}.png"
    by_author_entropy_png = out_dir / f"rq2_by_author_entropy_{run_id}.png"
    top_authors_png = out_dir / f"rq2_top_authors_{run_id}.png"

    _plot_by_post(post_stats, by_post_png)
    _plot_author_entropy(author_stats, by_author_entropy_png)
    _plot_top_authors(author_stats, top_authors_png)

    test_summary = pd.DataFrame(
        [
            {"group_level": "post_id", "h_stat": post_h, "p_value": post_p, "n_groups": post_k},
            {"group_level": "thread_id", "h_stat": thread_h, "p_value": thread_p, "n_groups": thread_k},
            {"group_level": "author_id", "h_stat": author_h, "p_value": author_p, "n_groups": author_k},
        ]
    )
    test_csv = out_dir / f"rq2_group_kruskal_{run_id}.csv"
    test_summary.to_csv(test_csv, index=False)

    findings = out_dir / f"rq2_group_findings_{run_id}.txt"
    findings.write_text(
        "\n".join(
            [
                f"Kruskal-Wallis post-level positive proportion: H={post_h:.4f}, p={post_p:.6g}",
                f"Kruskal-Wallis thread-level positive proportion: H={thread_h:.4f}, p={thread_p:.6g}",
                f"Kruskal-Wallis author-level positive proportion: H={author_h:.4f}, p={author_p:.6g}",
                "Pairwise Mann-Whitney U tests were run with Bonferroni correction on sufficiently sized groups.",
            ]
        ),
        encoding="utf-8",
    )

    manifest_path = _write_manifest(
        run_id,
        {"seed": args.seed, "input": input_path.as_posix(), "pairwise_min_n": 10},
        [
            post_csv,
            thread_csv,
            author_csv,
            test_csv,
            post_pairwise_csv,
            author_pairwise_csv,
            by_post_png,
            by_author_entropy_png,
            top_authors_png,
            findings,
        ],
    )

    print("RQ2 group analysis complete")
    print(f"run_id: {run_id}")
    print(f"post_kruskal_H: {post_h:.6f}")
    print(f"post_kruskal_p: {post_p:.6g}")
    print(f"manifest_path: {manifest_path}")


if __name__ == "__main__":
    main()
