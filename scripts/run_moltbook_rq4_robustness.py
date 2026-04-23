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
import yaml

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.figsize": (10, 6), "font.size": 12, "figure.dpi": 150})

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


def _label_from_score(score: float, threshold: float = 0.05) -> str:
    if score > threshold:
        return "positive"
    if score < -threshold:
        return "negative"
    return "neutral"


def _variant_frame(df: pd.DataFrame, variant: Dict[str, Any]) -> pd.DataFrame:
    text_field = variant["text_field"]
    min_words = int(variant.get("min_word_count", 1))
    work = df.copy()

    if text_field not in work.columns:
        raise ValueError(f"Variant {variant['id']} text field not found: {text_field}")

    words = work[text_field].fillna("").astype(str).str.split().str.len()
    work = work[words >= min_words].copy()

    scorer = str(variant.get("scorer", "ensemble"))
    if scorer == "ensemble":
        label_col = "ensemble_label"
    elif scorer == "vader":
        label_col = "vader_label"
    elif scorer == "sentiwordnet":
        label_col = "swn_label"
    else:
        raise ValueError(f"Unsupported scorer in variant {variant['id']}: {scorer}")

    work["variant_label"] = work[label_col].astype(str)
    return work


def _write_manifest(run_id: str, params: Dict[str, Any], outputs: List[Path]) -> Path:
    manifest_dir = Path("data/manifests")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    out = manifest_dir / f"rq4_robustness_manifest_{run_id}.json"
    out.write_text(
        json.dumps(
            {
                "experiment": "rq4_robustness_variants",
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
    parser = argparse.ArgumentParser(description="RQ4 robustness analysis across preprocessing/scoring variants.")
    parser.add_argument("--comments-csv", default="", help="Rule-based comments CSV; latest if empty")
    parser.add_argument("--preprocessed-csv", default="", help="Preprocessed CSV; latest if empty")
    parser.add_argument("--variants-yaml", default="configs/robustness_variants.yaml", help="Variants config")
    parser.add_argument("--figures-dir", default="data/figures", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--run-id", default="", help="Optional run ID")
    args = parser.parse_args()

    comments_path = Path(args.comments_csv) if args.comments_csv else _find_latest("data/rule_based/moltbook_rule_based_comments_*.csv")
    prep_path = Path(args.preprocessed_csv) if args.preprocessed_csv else _find_latest("data/preprocessed_rule_based/moltbook_preprocessed_rule_based_*.csv")
    variants_path = Path(args.variants_yaml)

    for p in [comments_path, prep_path, variants_path]:
        if not p.exists():
            raise FileNotFoundError(f"Input not found: {p}")

    run_id = args.run_id.strip() or _extract_run_id(comments_path)
    out_dir = Path(args.figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)

    comments = pd.read_csv(comments_path)
    prep = pd.read_csv(prep_path)
    variants = yaml.safe_load(variants_path.read_text(encoding="utf-8")).get("variants", [])

    merged = prep.merge(
        comments[["comment_id", "vader_label", "swn_label", "ensemble_label", "vader_compound"]],
        on="comment_id",
        how="inner",
    )

    rows: List[Dict[str, Any]] = []
    for var in variants:
        vdf = _variant_frame(merged, var)
        counts = vdf["variant_label"].value_counts().reindex(LABELS, fill_value=0)
        total = int(counts.sum())
        props = counts / max(total, 1)
        dominant = str(props.idxmax()) if total else "neutral"
        rows.append(
            {
                "variant_id": var["id"],
                "description": var.get("description", ""),
                "n_comments": total,
                "prop_neg": float(props["negative"]),
                "prop_neu": float(props["neutral"]),
                "prop_pos": float(props["positive"]),
                "dominant_class": dominant,
                "vader_mean_compound": float(vdf["vader_compound"].mean()) if total else 0.0,
            }
        )

    res = pd.DataFrame(rows)
    baseline_pos = float(res.loc[res["variant_id"] == "v1_baseline", "prop_pos"].iloc[0])
    res["delta_pos_from_baseline"] = (res["prop_pos"] - baseline_pos).astype(float)
    res["stable?"] = res["delta_pos_from_baseline"].abs() < 0.05

    cv_pos = float(res["prop_pos"].std() / max(res["prop_pos"].mean(), 1e-12))
    max_dev = float(res["delta_pos_from_baseline"].abs().max())

    matrix_cols = [
        "variant_id",
        "description",
        "n_comments",
        "prop_neg",
        "prop_neu",
        "prop_pos",
        "dominant_class",
        "delta_pos_from_baseline",
        "stable?",
    ]

    matrix_csv = out_dir / f"rq4_robustness_matrix_{run_id}.csv"
    res[matrix_cols].to_csv(matrix_csv, index=False)

    hm_data = res.set_index("variant_id")[["prop_neg", "prop_neu", "prop_pos"]]
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    sns.heatmap(hm_data, annot=True, fmt=".3f", cmap="coolwarm", vmin=0, vmax=1, ax=ax)
    ax.set_title("RQ4 Robustness Matrix")

    baseline_idx = list(hm_data.index).index("v1_baseline")
    ax.add_patch(plt.Rectangle((0, baseline_idx), 3, 1, fill=False, edgecolor="black", lw=2))

    fig.tight_layout()
    heatmap_png = out_dir / f"rq4_robustness_heatmap_{run_id}.png"
    fig.savefig(heatmap_png, dpi=150)
    plt.close(fig)

    all_neutral = bool((res["dominant_class"] == "neutral").all())
    sensitive = res.loc[res["dominant_class"] != "neutral", "variant_id"].tolist()
    verdict = "supported" if all_neutral else "partially supported"

    summary_path = out_dir / f"rq4_findings_summary_{run_id}.txt"
    summary_path.write_text(
        "\n".join(
            [
                "HYPOTHESIS VERDICT",
                verdict.upper(),
                f"cv_pos={cv_pos:.6f}",
                f"max_abs_delta_pos={max_dev:.6f}",
                f"dominant_class_all_variants_neutral={all_neutral}",
                f"sensitive_variants={sensitive if sensitive else 'none'}",
            ]
        ),
        encoding="utf-8",
    )

    manifest_path = _write_manifest(
        run_id,
        {
            "seed": args.seed,
            "comments_csv": comments_path.as_posix(),
            "preprocessed_csv": prep_path.as_posix(),
            "variants_yaml": variants_path.as_posix(),
            "stability_threshold": 0.05,
        },
        [matrix_csv, heatmap_png, summary_path],
    )

    print("RQ4 robustness analysis complete")
    print(f"run_id: {run_id}")
    print(f"cv_positive: {cv_pos:.6f}")
    print(f"max_abs_delta_positive: {max_dev:.6f}")
    print(f"dominant_class_neutral_all: {all_neutral}")
    print(f"manifest_path: {manifest_path}")


if __name__ == "__main__":
    main()
