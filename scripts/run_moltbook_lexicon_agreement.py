from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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


def _mcnemar_exact(n01: int, n10: int) -> float:
    discordant = n01 + n10
    if discordant == 0:
        return 1.0
    k = min(n01, n10)
    two_sided = 2.0 * sum(math.comb(discordant, i) for i in range(0, k + 1)) / (2.0 ** discordant)
    return float(min(1.0, two_sided))


def _normalize_label(x: Any) -> str:
    t = str(x or "").strip().lower()
    return t if t in LABELS else ""


def _write_manifest(run_id: str, params: Dict[str, Any], outputs: List[Path]) -> Path:
    manifest_dir = Path("data/manifests")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    out = manifest_dir / f"rq2_lexicon_agreement_manifest_{run_id}.json"
    out.write_text(
        json.dumps(
            {
                "experiment": "rq2_lexicon_agreement",
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
    parser = argparse.ArgumentParser(description="RQ2/RQ4 lexicon agreement analysis.")
    parser.add_argument("--comments-csv", default="", help="Rule-based comments CSV; latest if omitted")
    parser.add_argument("--gold-csv", default="", help="Gold set CSV; latest if omitted")
    parser.add_argument("--figures-dir", default="data/figures", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--run-id", default="", help="Optional run id")
    args = parser.parse_args()

    comments_path = Path(args.comments_csv) if args.comments_csv else _find_latest("data/rule_based/moltbook_rule_based_comments_*.csv")
    gold_path = Path(args.gold_csv) if args.gold_csv else _find_latest("data/gold/moltbook_goldset_sample_*.csv")

    if not comments_path.exists():
        raise FileNotFoundError(f"Comments CSV not found: {comments_path}")

    out_dir = Path(args.figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = args.run_id.strip() or _extract_run_id(comments_path)
    np.random.seed(args.seed)

    df = pd.read_csv(comments_path)
    for col in ["vader_label", "swn_label"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
        df[col] = df[col].map(_normalize_label)

    conf = pd.crosstab(df["vader_label"], df["swn_label"]).reindex(index=LABELS, columns=LABELS, fill_value=0)
    row_norm = conf.div(conf.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    long_rows: List[Dict[str, Any]] = []
    for r in LABELS:
        for c in LABELS:
            long_rows.append(
                {
                    "vader_label": r,
                    "swn_label": c,
                    "count": int(conf.loc[r, c]),
                    "row_prop": float(row_norm.loc[r, c]),
                }
            )
    conf_csv = out_dir / f"rq2_lexicon_agreement_matrix_{run_id}.csv"
    pd.DataFrame(long_rows).to_csv(conf_csv, index=False)

    off = conf.copy()
    for l in LABELS:
        off.loc[l, l] = 0
    idx = np.unravel_index(np.argmax(off.to_numpy()), off.shape)
    top_pattern = {"vader": LABELS[idx[0]], "swn": LABELS[idx[1]], "count": int(off.iloc[idx[0], idx[1]])}

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    annot = conf.astype(str) + "\n(" + row_norm.round(3).astype(str) + ")"
    sns.heatmap(conf, annot=annot, fmt="", cmap="Blues", cbar=True, ax=ax)
    ax.set_title("RQ2 Lexicon Agreement: VADER vs SentiWordNet")
    ax.set_xlabel("SentiWordNet label")
    ax.set_ylabel("VADER label")
    fig.tight_layout()
    heatmap_png = out_dir / f"rq2_lexicon_agreement_heatmap_{run_id}.png"
    fig.savefig(heatmap_png, dpi=150)
    plt.close(fig)

    mcnemar_note = "No human labels available; reported descriptive disagreement pattern only."
    mcnemar_payload: Dict[str, Any] = {}

    if gold_path.exists():
        gold = pd.read_csv(gold_path)
        if "adjudicated_label" in gold.columns:
            gold["adjudicated_label"] = gold["adjudicated_label"].map(_normalize_label)
            valid = gold["adjudicated_label"].isin(LABELS)
            if valid.any():
                gb = gold.loc[valid].copy()
                gb["vader_label"] = gb["vader_label"].map(_normalize_label)
                gb["swn_label"] = gb["swn_label"].map(_normalize_label)
                gb = gb[gb["vader_label"].isin(LABELS) & gb["swn_label"].isin(LABELS)]

                vader_correct = (gb["vader_label"] == gb["adjudicated_label"]).to_numpy()
                swn_correct = (gb["swn_label"] == gb["adjudicated_label"]).to_numpy()
                n01 = int((~vader_correct & swn_correct).sum())
                n10 = int((vader_correct & ~swn_correct).sum())
                p_exact = _mcnemar_exact(n01, n10)
                mcnemar_payload = {"n01": n01, "n10": n10, "p_value_exact": p_exact}
                mcnemar_note = "McNemar exact test computed against adjudicated human labels."

    summary = {
        "run_id": run_id,
        "top_disagreement_pattern": top_pattern,
        "mcnemar_note": mcnemar_note,
        "mcnemar": mcnemar_payload,
    }
    summary_path = out_dir / f"rq2_lexicon_agreement_summary_{run_id}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    manifest_path = _write_manifest(
        run_id,
        {
            "seed": args.seed,
            "comments_csv": comments_path.as_posix(),
            "gold_csv": gold_path.as_posix() if gold_path.exists() else "",
        },
        [conf_csv, heatmap_png, summary_path],
    )

    print("Lexicon agreement analysis complete")
    print(f"run_id: {run_id}")
    print(f"top_disagreement_pattern: {top_pattern}")
    print(f"mcnemar_note: {mcnemar_note}")
    print(f"manifest_path: {manifest_path}")


if __name__ == "__main__":
    main()
