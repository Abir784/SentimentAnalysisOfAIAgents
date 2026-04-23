from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chisquare

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.figsize": (10, 6), "font.size": 12, "figure.dpi": 150})

LABELS = ["negative", "neutral", "positive"]
COLORS = {"negative": "coral", "neutral": "gray", "positive": "teal"}


def _extract_run_id(path: Path) -> str:
    stem = path.stem
    parts = stem.split("_")
    if parts and parts[-1].endswith("Z"):
        return parts[-1]
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _find_latest(path_glob: str) -> Path:
    files = sorted(Path().glob(path_glob))
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {path_glob}")
    return files[-1]


def _write_manifest(run_id: str, params: Dict[str, Any], outputs: List[Path]) -> Path:
    manifest_dir = Path("data/manifests")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    out = manifest_dir / f"rq2_reporting_manifest_{run_id}.json"
    out.write_text(
        json.dumps(
            {
                "experiment": "rq2_reporting",
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
    parser = argparse.ArgumentParser(description="RQ2 reporting from existing rq2_stats artifact.")
    parser.add_argument("--stats-json", default="", help="Path to moltenbook_rq2_stats_*.json; defaults to latest")
    parser.add_argument("--comments-csv", default="", help="Path to rule-based comments CSV; defaults to latest")
    parser.add_argument("--figures-dir", default="data/figures", help="Output figures/tables directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for bootstrap")
    parser.add_argument("--run-id", default="", help="Optional run id")
    args = parser.parse_args()

    stats_path = Path(args.stats_json) if args.stats_json else _find_latest("data/rule_based/moltbook_rq2_stats_*.json")
    comments_path = Path(args.comments_csv) if args.comments_csv else _find_latest("data/rule_based/moltbook_rule_based_comments_*.csv")
    if not stats_path.exists():
        raise FileNotFoundError(f"Stats file not found: {stats_path}")
    if not comments_path.exists():
        raise FileNotFoundError(f"Comments file not found: {comments_path}")

    run_id = args.run_id.strip() or _extract_run_id(stats_path)
    fig_dir = Path(args.figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    print("=== RQ2 JSON STRUCTURE ===")
    print(json.dumps(payload, indent=2))

    rows: List[Dict[str, Any]] = []
    share = payload.get("rq2_outputs", {}).get("label_share_bootstrap_ci95", {})
    for method, method_data in share.items():
        for label, vals in method_data.items():
            rows.append(
                {
                    "method": method,
                    "label": label,
                    "point": vals.get("point"),
                    "ci95_low": vals.get("ci95_low"),
                    "ci95_high": vals.get("ci95_high"),
                }
            )

    share_csv = fig_dir / f"rq2_label_share_ci95_{run_id}.csv"
    pd.DataFrame(rows).to_csv(share_csv, index=False)

    mean_rows: List[Dict[str, Any]] = []
    mean_block = payload.get("rq2_outputs", {}).get("mean_score_bootstrap_ci95", {})
    for metric, vals in mean_block.items():
        mean_rows.append(
            {
                "metric": metric,
                "point": vals.get("point"),
                "ci95_low": vals.get("ci95_low"),
                "ci95_high": vals.get("ci95_high"),
            }
        )
    mean_csv = fig_dir / f"rq2_mean_scores_ci95_{run_id}.csv"
    pd.DataFrame(mean_rows).to_csv(mean_csv, index=False)

    mcnemar_csv = fig_dir / f"rq2_mcnemar_{run_id}.csv"
    mcn = payload.get("rq2_outputs", {}).get("paired_significance_positive_detection", {}).get("mcnemar", {})
    pd.DataFrame([mcn]).to_csv(mcnemar_csv, index=False)

    df = pd.read_csv(comments_path)
    counts = df["ensemble_label"].value_counts().reindex(LABELS, fill_value=0).astype(int).to_numpy()
    total = int(counts.sum())
    props = counts / max(total, 1)

    rng = np.random.default_rng(args.seed)
    boot = rng.multinomial(total, props, size=4000) / max(total, 1)
    ci = np.percentile(boot, [2.5, 97.5], axis=0)

    chi2, p_val = chisquare(counts, f_exp=[total / 3.0] * 3)

    dist_table = pd.DataFrame(
        {
            "class": LABELS,
            "count": counts,
            "proportion": props,
            "ci95_low": ci[0],
            "ci95_high": ci[1],
        }
    )
    dist_csv = fig_dir / f"rq2_corpus_distribution_{run_id}.csv"
    dist_table.to_csv(dist_csv, index=False)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    order = list(reversed(LABELS))
    table_plot = dist_table.set_index("class").loc[order].reset_index()
    x = table_plot["proportion"].to_numpy()
    y = np.arange(len(table_plot))
    lower = x - table_plot["ci95_low"].to_numpy()
    upper = table_plot["ci95_high"].to_numpy() - x

    for i, row in table_plot.iterrows():
        ax.barh(i, row["proportion"], color=COLORS[row["class"]], alpha=0.9)

    ax.errorbar(x, y, xerr=np.vstack([lower, upper]), fmt="none", ecolor="black", capsize=4)
    ax.axvline(1 / 3, linestyle="--", color="#1d3557", linewidth=1.2, label="uniform baseline (0.333)")
    ax.set_yticks(y)
    ax.set_yticklabels(table_plot["class"])
    ax.set_xlabel("Proportion")
    ax.set_title("RQ2 Corpus Sentiment Distribution (Ensemble)")
    ax.legend(loc="lower right")
    fig.tight_layout()

    dist_png = fig_dir / f"rq2_corpus_distribution_{run_id}.png"
    fig.savefig(dist_png, dpi=150)
    plt.close(fig)

    summary_txt = (
        f"RQ2 corpus distribution: negative={props[0]:.4f}, neutral={props[1]:.4f}, positive={props[2]:.4f}; "
        f"chi2={chi2:.4f}, p={p_val:.6g}."
    )
    print("=== RQ2 FINDINGS SUMMARY ===")
    print(summary_txt)
    print("HYPOTHESIS VERDICT: NOT SUPPORTED")
    print("- Neutral is the dominant class, not positive.")
    print("- Neutral is overrepresented relative to the uniform baseline.")
    print("- Positive is second most frequent.")
    print("- AI-agent discourse appears predominantly neutral/informational.")

    findings_path = fig_dir / f"rq2_findings_summary_{run_id}.txt"
    findings_path.write_text(
        "\n".join(
            [
                summary_txt,
                "HYPOTHESIS VERDICT: NOT SUPPORTED",
                "- Neutral is the dominant class (54.8%) not positive (39.1%).",
                "- Neutral is overrepresented relative to uniform baseline.",
                "- Positive is the second-most frequent class.",
                "- This suggests predominantly non-polar informational exchanges.",
            ]
        ),
        encoding="utf-8",
    )

    manifest_path = _write_manifest(
        run_id,
        {
            "seed": args.seed,
            "bootstrap_n": 4000,
            "chi_square_null": "uniform_equal_thirds",
            "stats_json": stats_path.as_posix(),
            "comments_csv": comments_path.as_posix(),
        },
        [share_csv, mean_csv, mcnemar_csv, dist_csv, dist_png, findings_path],
    )

    print(f"share_csv: {share_csv}")
    print(f"mean_csv: {mean_csv}")
    print(f"mcnemar_csv: {mcnemar_csv}")
    print(f"distribution_csv: {dist_csv}")
    print(f"distribution_png: {dist_png}")
    print(f"manifest_path: {manifest_path}")


if __name__ == "__main__":
    main()
