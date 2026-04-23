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
from scipy.stats import chi2_contingency, kruskal, spearmanr

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.figsize": (10, 6), "font.size": 12, "figure.dpi": 150})

LABELS = ["negative", "neutral", "positive"]
KEY_FEATURES = [
    "text_len_words",
    "thread_depth",
    "upvotes",
    "polarity_compound",
    "n_siblings",
    "author_comment_count",
]


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


def _eta_squared_kw(h: float, n: int, k: int) -> float:
    if n <= k:
        return 0.0
    return float(max(0.0, (h - k + 1) / (n - k)))


def _bootstrap_ci(values: np.ndarray, fn, n_boot: int, seed: int) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(fn(values[idx]))
    lo, hi = np.percentile(np.array(boots, dtype=float), [2.5, 97.5])
    return float(lo), float(hi)


def _prepare_df(rule_path: Path, features_path: Path, staged_path: Path) -> pd.DataFrame:
    rb = pd.read_csv(rule_path)
    feat = pd.read_csv(features_path)
    staged = pd.DataFrame(_read_jsonl(staged_path))

    for col in ["comment_id", "thread_id", "author_id", "post_id"]:
        if col in rb.columns:
            rb[col] = rb[col].astype(str)
        if col in feat.columns:
            feat[col] = feat[col].astype(str)
        if col in staged.columns:
            staged[col] = staged[col].astype(str)

    use_feat_cols = [
        c
        for c in ["comment_id", "token_count", "char_count", "question_count", "exclamation_count"]
        if c in feat.columns
    ]
    merged = rb.merge(feat[use_feat_cols], on="comment_id", how="left")

    staged_small_cols = [c for c in ["comment_id", "thread_id", "author_id", "post_id", "level", "upvotes", "is_verified"] if c in staged.columns]
    merged = merged.merge(staged[staged_small_cols], on="comment_id", how="left", suffixes=("", "_staged"))

    merged["thread_id"] = merged.get("thread_id", merged.get("thread_id_staged", "")).fillna("").astype(str)
    merged["author_id"] = merged.get("author_id", merged.get("author_id_staged", "")).fillna("").astype(str)
    merged["post_id"] = merged.get("post_id", merged.get("post_id_staged", "")).fillna("").astype(str)

    merged["text_len_words"] = merged.get("token_count", pd.Series([np.nan] * len(merged))).fillna(
        merged.get("text", "").fillna("").astype(str).str.split().str.len()
    )
    merged["text_len_chars"] = merged.get("char_count", pd.Series([np.nan] * len(merged))).fillna(
        merged.get("text", "").fillna("").astype(str).str.len()
    )
    merged["thread_depth"] = pd.to_numeric(merged.get("level", 0), errors="coerce").fillna(0)
    if merged["thread_depth"].nunique(dropna=True) <= 1:
        # Fallback when staged level metadata is absent/flat: use within-thread order depth proxy.
        merged["thread_depth"] = merged.groupby("thread_id").cumcount().astype(float)
    merged["upvotes"] = pd.to_numeric(merged.get("upvotes", 0), errors="coerce").fillna(0)
    merged["is_verified"] = merged.get("is_verified", False).astype(str).str.lower().isin(["1", "true", "yes"])
    merged["n_siblings"] = merged.groupby("thread_id")["comment_id"].transform("count") - 1
    merged["author_comment_count"] = merged.groupby("author_id")["comment_id"].transform("count")
    merged["polarity_compound"] = pd.to_numeric(merged.get("vader_compound", 0.0), errors="coerce").fillna(0.0)
    merged["has_question_mark"] = merged.get("question_count", merged.get("text", "").fillna("").astype(str).str.contains(r"\?", regex=True)).fillna(0)
    merged["has_question_mark"] = merged["has_question_mark"].astype(float).gt(0).astype(int)
    merged["has_exclamation"] = merged.get("exclamation_count", merged.get("text", "").fillna("").astype(str).str.contains("!")).fillna(0)
    merged["has_exclamation"] = merged["has_exclamation"].astype(float).gt(0).astype(int)

    return merged


def _kruskal_rows(df: pd.DataFrame, features: List[str], seed: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    n_boot = 500
    for i, feat in enumerate(features):
        vals = [df[df["ensemble_label"] == c][feat].dropna().to_numpy(dtype=float) for c in LABELS]
        if any(v.size == 0 for v in vals):
            rows.append(
                {
                    "feature": feat,
                    "test": "kruskal",
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "effect_size": np.nan,
                    "effect_name": "eta_squared",
                    "ci95_low": np.nan,
                    "ci95_high": np.nan,
                }
            )
            continue

        try:
            h, p = kruskal(*vals)
            eta2 = _eta_squared_kw(float(h), int(sum(v.size for v in vals)), 3)
        except ValueError:
            h, p, eta2 = 0.0, 1.0, 0.0

        pooled = np.concatenate(vals)

        def _boot_eta(sample: np.ndarray) -> float:
            n = len(sample)
            idx = np.random.default_rng(seed + i).integers(0, n, size=n)
            ss = sample[idx]
            thirds = np.array_split(ss, 3)
            try:
                h2, _ = kruskal(*thirds)
                return _eta_squared_kw(float(h2), n, 3)
            except ValueError:
                return 0.0

        ci_lo, ci_hi = _bootstrap_ci(pooled, _boot_eta, n_boot=n_boot, seed=seed + i + 100)
        rows.append(
            {
                "feature": feat,
                "test": "kruskal",
                "statistic": float(h),
                "p_value": float(p),
                "effect_size": float(eta2),
                "effect_name": "eta_squared",
                "ci95_low": float(ci_lo),
                "ci95_high": float(ci_hi),
            }
        )
    return pd.DataFrame(rows)


def _chi_square_rows(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for feat in features:
        tab = pd.crosstab(df[feat], df["ensemble_label"]).reindex(columns=LABELS, fill_value=0)
        chi2, p, _, _ = chi2_contingency(tab)
        n = tab.to_numpy().sum()
        r, k = tab.shape
        denom = min(r - 1, k - 1)
        cramer_v = float(np.sqrt((chi2 / max(n, 1)) / max(denom, 1)))
        rows.append(
            {
                "feature": feat,
                "test": "chi_square",
                "statistic": float(chi2),
                "p_value": float(p),
                "effect_size": cramer_v,
                "effect_name": "cramers_v",
                "ci95_low": np.nan,
                "ci95_high": np.nan,
            }
        )
    return pd.DataFrame(rows)


def _plot_box_grid(df: pd.DataFrame, stats_df: pd.DataFrame, out_path: Path) -> None:
    palette = {"negative": "#e76f51", "neutral": "#9e9e9e", "positive": "#2a9d8f"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=150)
    axes = axes.flatten()
    for ax, feat in zip(axes, KEY_FEATURES):
        sns.boxplot(data=df, x="ensemble_label", y=feat, order=LABELS, palette=palette, ax=ax)
        p_vals = stats_df[(stats_df["feature"] == feat) & (stats_df["test"] == "kruskal")]["p_value"]
        p_text = f"p={p_vals.iloc[0]:.4g}" if len(p_vals) else "p=NA"
        ax.set_title(f"{feat} ({p_text})")
        ax.set_xlabel("sentiment class")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_variability_scatter(thread_stats: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    sc = ax.scatter(
        thread_stats["max_depth"],
        thread_stats["sentiment_std"],
        c=thread_stats["n_comments"],
        cmap="viridis",
        alpha=0.85,
    )
    try:
        sns.regplot(
            data=thread_stats,
            x="max_depth",
            y="sentiment_std",
            lowess=True,
            scatter=False,
            color="#d62828",
            ax=ax,
        )
    except Exception:
        sns.regplot(data=thread_stats, x="max_depth", y="sentiment_std", scatter=False, color="#d62828", ax=ax)

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("n_comments")
    ax.set_title("RQ3 Variability vs Thread Depth")
    ax.set_xlabel("max_thread_depth")
    ax.set_ylabel("sentiment_std")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_verified(df: pd.DataFrame, out_path: Path) -> None:
    group = (
        df.groupby(["is_verified", "ensemble_label"]).size().rename("count").reset_index()
    )
    total = group.groupby("is_verified")["count"].transform("sum")
    group["prop"] = group["count"] / total

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    sns.barplot(data=group, x="is_verified", y="prop", hue="ensemble_label", hue_order=LABELS, ax=ax)
    ax.set_title("RQ3 Sentiment by Verification Status")
    ax.set_xlabel("is_verified")
    ax.set_ylabel("proportion")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _write_manifest(run_id: str, params: Dict[str, Any], outputs: List[Path]) -> Path:
    manifest_dir = Path("data/manifests")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    out = manifest_dir / f"rq3_feature_analysis_manifest_{run_id}.json"
    out.write_text(
        json.dumps(
            {
                "experiment": "rq3_feature_analysis",
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
    parser = argparse.ArgumentParser(description="RQ3 feature-sentiment association and variability analysis.")
    parser.add_argument("--rule-based-csv", default="", help="Rule-based comments CSV; latest if empty")
    parser.add_argument("--features-csv", default="", help="Features CSV; latest if empty")
    parser.add_argument("--staged-jsonl", default="data/staged/moltbook_comments_all.jsonl", help="Staged comments JSONL")
    parser.add_argument("--figures-dir", default="data/figures", help="Output figures directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--run-id", default="", help="Optional run ID")
    args = parser.parse_args()

    rb_path = Path(args.rule_based_csv) if args.rule_based_csv else _find_latest("data/rule_based/moltbook_rule_based_comments_*.csv")
    feat_path = Path(args.features_csv) if args.features_csv else _find_latest("data/features_rule_based/moltbook_features_rule_based_*.csv")
    staged_path = Path(args.staged_jsonl)

    for p in [rb_path, feat_path, staged_path]:
        if not p.exists():
            raise FileNotFoundError(f"Input not found: {p}")

    run_id = args.run_id.strip() or _extract_run_id(rb_path)
    out_dir = Path(args.figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _prepare_df(rb_path, feat_path, staged_path)
    df = df[df["ensemble_label"].isin(LABELS)].copy()

    features = [
        "text_len_words",
        "text_len_chars",
        "thread_depth",
        "upvotes",
        "n_siblings",
        "author_comment_count",
        "polarity_compound",
    ]
    binary_features = ["is_verified", "has_question_mark", "has_exclamation"]

    desc = df.groupby("ensemble_label")[features + binary_features].agg(["mean", "median", "std", "count"])
    desc_csv = out_dir / f"rq3_feature_descriptive_{run_id}.csv"
    desc.to_csv(desc_csv)

    kr_df = _kruskal_rows(df, features, args.seed)
    chi_df = _chi_square_rows(df, binary_features)
    stats_df = pd.concat([kr_df, chi_df], ignore_index=True)
    stats_csv = out_dir / f"rq3_feature_stats_{run_id}.csv"
    stats_df.to_csv(stats_csv, index=False)

    thread_stats = (
        df.groupby("thread_id")
        .agg(
            sentiment_std=("polarity_compound", "std"),
            mean_depth=("thread_depth", "mean"),
            max_depth=("thread_depth", "max"),
            mean_len=("text_len_words", "mean"),
            n_comments=("comment_id", "count"),
        )
        .dropna()
        .reset_index()
    )

    r_depth, p_depth = spearmanr(thread_stats["max_depth"], thread_stats["sentiment_std"])
    r_len, p_len = spearmanr(thread_stats["mean_len"], thread_stats["sentiment_std"])

    def boot_spearman(arr_x: np.ndarray, arr_y: np.ndarray, seed: int) -> Tuple[float, float]:
        rng = np.random.default_rng(seed)
        n = len(arr_x)
        vals = []
        for _ in range(1000):
            idx = rng.integers(0, n, size=n)
            r, _ = spearmanr(arr_x[idx], arr_y[idx])
            vals.append(0.0 if np.isnan(r) else float(r))
        lo, hi = np.percentile(vals, [2.5, 97.5])
        return float(lo), float(hi)

    depth_ci = boot_spearman(thread_stats["max_depth"].to_numpy(), thread_stats["sentiment_std"].to_numpy(), args.seed)
    len_ci = boot_spearman(thread_stats["mean_len"].to_numpy(), thread_stats["sentiment_std"].to_numpy(), args.seed + 1)

    corr_csv = out_dir / f"rq3_variability_spearman_{run_id}.csv"
    pd.DataFrame(
        [
            {
                "relation": "max_depth_vs_sentiment_std",
                "spearman_r": float(r_depth),
                "p_value": float(p_depth),
                "ci95_low": depth_ci[0],
                "ci95_high": depth_ci[1],
            },
            {
                "relation": "mean_len_vs_sentiment_std",
                "spearman_r": float(r_len),
                "p_value": float(p_len),
                "ci95_low": len_ci[0],
                "ci95_high": len_ci[1],
            },
        ]
    ).to_csv(corr_csv, index=False)

    box_png = out_dir / f"rq3_feature_boxplots_{run_id}.png"
    scatter_png = out_dir / f"rq3_variability_scatter_{run_id}.png"
    verified_png = out_dir / f"rq3_verified_vs_unverified_{run_id}.png"

    _plot_box_grid(df, stats_df, box_png)
    _plot_variability_scatter(thread_stats, scatter_png)
    _plot_verified(df, verified_png)

    verdict = "not supported"
    if (r_depth > 0.1 and p_depth < 0.05) and (r_len > 0.1 and p_len < 0.05):
        verdict = "supported"
    elif (r_depth > 0.1 and p_depth < 0.05) or (r_len > 0.1 and p_len < 0.05):
        verdict = "partially supported"

    verdict_path = out_dir / f"rq3_hypothesis_verdict_{run_id}.txt"
    verdict_path.write_text(
        "\n".join(
            [
                "HYPOTHESIS VERDICT",
                verdict.upper(),
                f"depth vs variability: r={r_depth:.4f}, p={p_depth:.6g}, ci95=[{depth_ci[0]:.4f}, {depth_ci[1]:.4f}]",
                f"length vs variability: r={r_len:.4f}, p={p_len:.6g}, ci95=[{len_ci[0]:.4f}, {len_ci[1]:.4f}]",
            ]
        ),
        encoding="utf-8",
    )

    manifest_path = _write_manifest(
        run_id,
        {
            "seed": args.seed,
            "rule_based_csv": rb_path.as_posix(),
            "features_csv": feat_path.as_posix(),
            "staged_jsonl": staged_path.as_posix(),
            "bootstrap_n": 4000,
        },
        [desc_csv, stats_csv, corr_csv, box_png, scatter_png, verified_png, verdict_path],
    )

    print("RQ3 feature analysis complete")
    print(f"run_id: {run_id}")
    print(f"spearman_depth_r: {r_depth:.6f}")
    print(f"spearman_depth_p: {p_depth:.6g}")
    print(f"spearman_len_r: {r_len:.6f}")
    print(f"spearman_len_p: {p_len:.6g}")
    print(f"hypothesis_verdict: {verdict}")
    print(f"manifest_path: {manifest_path}")


if __name__ == "__main__":
    main()
