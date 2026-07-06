from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

LABELS = ["negative", "neutral", "positive"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}


def _cohen_kappa(y1: np.ndarray, y2: np.ndarray, n_labels: int) -> float:
    conf = np.zeros((n_labels, n_labels), dtype=float)
    for a, b in zip(y1, y2):
        conf[a, b] += 1

    n = conf.sum()
    if n == 0:
        return float("nan")

    p0 = np.trace(conf) / n
    pe = (conf.sum(axis=1) * conf.sum(axis=0)).sum() / (n * n)
    if pe >= 1.0:
        return 1.0
    return float((p0 - pe) / (1.0 - pe))


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_labels: int) -> float:
    f1_scores: List[float] = []
    for i in range(n_labels):
        tp = float(((y_true == i) & (y_pred == i)).sum())
        fp = float(((y_true != i) & (y_pred == i)).sum())
        fn = float(((y_true == i) & (y_pred != i)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        f1_scores.append(f1)

    return float(np.mean(f1_scores))


def _normalize_label(value: object) -> str:
    text = str(value or "").strip().lower()
    if text not in LABEL_TO_ID:
        return ""
    return text


def _majority_human_label(df: pd.DataFrame) -> pd.Series:
    r1 = df["rater_1_label"].map(_normalize_label)
    r2 = df["rater_2_label"].map(_normalize_label)
    adjud = df["adjudicated_label"].map(_normalize_label)

    resolved = []
    for a, b, c in zip(r1, r2, adjud):
        if c in LABEL_TO_ID:
            resolved.append(c)
        elif a in LABEL_TO_ID and b in LABEL_TO_ID and a == b:
            resolved.append(a)
        else:
            resolved.append("")

    return pd.Series(resolved, index=df.index)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate two-rater gold-set annotation quality and method macro-F1."
    )
    parser.add_argument("--input", required=True, help="Annotated gold-set CSV path.")
    parser.add_argument(
        "--output",
        default="",
        help="Optional output JSON path. Defaults next to input file.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    required = [
        "rater_1_label",
        "rater_2_label",
        "adjudicated_label",
        "vader_label",
        "swn_label",
        "ensemble_label",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    r1 = df["rater_1_label"].map(_normalize_label)
    r2 = df["rater_2_label"].map(_normalize_label)
    paired_mask = r1.isin(LABEL_TO_ID) & r2.isin(LABEL_TO_ID)

    y1 = r1[paired_mask].map(LABEL_TO_ID).to_numpy(dtype=int)
    y2 = r2[paired_mask].map(LABEL_TO_ID).to_numpy(dtype=int)

    kappa = _cohen_kappa(y1, y2, len(LABELS)) if len(y1) else float("nan")

    human = _majority_human_label(df)
    human_mask = human.isin(LABEL_TO_ID)
    y_true = human[human_mask].map(LABEL_TO_ID).to_numpy(dtype=int)

    method_scores: Dict[str, float] = {}
    for col in ["vader_label", "swn_label", "ensemble_label"]:
        pred = df.loc[human_mask, col].map(_normalize_label)
        valid = pred.isin(LABEL_TO_ID)
        if valid.sum() == 0:
            method_scores[col] = float("nan")
            continue
        y_pred = pred[valid].map(LABEL_TO_ID).to_numpy(dtype=int)
        y_eval = y_true[valid.to_numpy()]
        method_scores[col] = _macro_f1(y_eval, y_pred, len(LABELS))

    output = {
        "input_file": input_path.as_posix(),
        "rows": int(len(df)),
        "paired_rows_for_kappa": int(len(y1)),
        "rows_with_resolved_human_label": int(human_mask.sum()),
        "cohen_kappa_rater1_rater2": kappa,
        "macro_f1_against_human": method_scores,
        "label_space": LABELS,
    }

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = input_path.with_name(input_path.stem + "_evaluation.json")

    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)


if __name__ == "__main__":
    main()
