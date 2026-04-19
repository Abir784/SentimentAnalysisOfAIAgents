from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipelines.polarity_moltbook import preprocess_for_sentiment
from src.utils.file_management import cleanup_old_files

TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")


def _ensure_nltk() -> Tuple[Any, Any, Any, Any]:
    import nltk
    from nltk.corpus import sentiwordnet as swn
    from nltk.corpus import wordnet as wn
    from nltk.stem import WordNetLemmatizer
    from nltk.tag import pos_tag

    resources = [
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
        ("corpora/sentiwordnet", "sentiwordnet"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ]
    for resource_path, resource_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(resource_name, quiet=True)

    return swn, wn, WordNetLemmatizer(), pos_tag


def _wordnet_pos(tag: str) -> str | None:
    if tag.startswith("J"):
        return "a"
    if tag.startswith("V"):
        return "v"
    if tag.startswith("N"):
        return "n"
    if tag.startswith("R"):
        return "r"
    return None


def _label_from_score(value: float, threshold: float) -> str:
    if value >= threshold:
        return "positive"
    if value <= -threshold:
        return "negative"
    return "neutral"


def _sentiwordnet_score(text: str, swn: Any, lemmatizer: Any, pos_tag: Any) -> float:
    cleaned = preprocess_for_sentiment(str(text or "")).lower()
    tokens = TOKEN_RE.findall(cleaned)
    if not tokens:
        return 0.0

    tagged = pos_tag(tokens)
    values: List[float] = []

    for token, tag in tagged:
        wn_pos = _wordnet_pos(tag)
        if wn_pos is None:
            continue
        lemma = lemmatizer.lemmatize(token, wn_pos)
        synsets = list(swn.senti_synsets(lemma, wn_pos))
        if not synsets:
            continue
        syn = synsets[0]
        values.append(float(syn.pos_score() - syn.neg_score()))

    if not values:
        return 0.0

    return float(sum(values) / len(values))


def _plot_label_shares(df: pd.DataFrame, out_path: Path) -> None:
    methods = [
        ("vader_label", "VADER"),
        ("swn_label", "SentiWordNet"),
        ("ensemble_label", "Ensemble"),
    ]

    rows: List[Dict[str, Any]] = []
    for col, name in methods:
        if col not in df.columns:
            continue
        shares = df[col].value_counts(normalize=True)
        for label, share in shares.items():
            rows.append({"method": name, "label": str(label), "share": float(share)})

    plot_df = pd.DataFrame(rows)
    plt.figure(figsize=(10, 5.5))
    if plot_df.empty:
        plt.text(0.5, 0.5, "No rule-based label data", ha="center", va="center")
        plt.axis("off")
    else:
        labels = ["negative", "neutral", "positive"]
        methods_order = ["VADER", "SentiWordNet", "Ensemble"]
        width = 0.24
        x = range(len(labels))

        for i, method in enumerate(methods_order):
            vals = []
            for lbl in labels:
                row = plot_df[(plot_df["method"] == method) & (plot_df["label"] == lbl)]
                vals.append(float(row["share"].iloc[0]) if not row.empty else 0.0)
            offset = (i - 1) * width
            plt.bar([v + offset for v in x], vals, width=width, label=method)

        plt.xticks(list(x), labels)
        plt.ylim(0, 1)
        plt.ylabel("Share")
        plt.title("Rule-Based Sentiment Label Share by Method")
        plt.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_score_distributions(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(10, 5.5))
    if df.empty:
        plt.text(0.5, 0.5, "No scored rows", ha="center", va="center")
        plt.axis("off")
    else:
        plt.hist(df["vader_compound"], bins=40, alpha=0.55, label="VADER compound")
        plt.hist(df["swn_score"], bins=40, alpha=0.55, label="SentiWordNet score")
        plt.title("Rule-Based Score Distributions")
        plt.xlabel("Score")
        plt.ylabel("Comments")
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run rule-based sentiment analysis with VADER and SentiWordNet."
    )
    parser.add_argument(
        "--input",
        default="",
        help="Input file (.csv or .jsonl). Defaults to latest preprocessed_rule_based CSV, else staged JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/rule_based",
        help="Output directory for rule-based artifacts.",
    )
    parser.add_argument(
        "--vader-threshold",
        type=float,
        default=0.05,
        help="Threshold for VADER label mapping.",
    )
    parser.add_argument(
        "--swn-threshold",
        type=float,
        default=0.02,
        help="Threshold for SentiWordNet label mapping.",
    )
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else Path("")
    if not args.input:
        preprocessed_candidates = sorted(
            Path("data/preprocessed_rule_based").glob("moltbook_preprocessed_rule_based_*.csv")
        )
        if preprocessed_candidates:
            input_path = preprocessed_candidates[-1]
        else:
            input_path = Path("data/staged/moltbook_comments_all.jsonl")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    if input_path.suffix.lower() == ".csv":
        df = pd.read_csv(input_path)
    else:
        rows: List[Dict[str, Any]] = []
        with input_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("Input file contains no rows.")

    for col in ["comment_id", "post_id", "thread_id", "author_id", "text", "is_verified", "upvotes"]:
        if col not in df.columns:
            df[col] = "" if col != "is_verified" else False

    df["text"] = df["text"].fillna("").astype(str)
    df = df[df["text"].str.strip().ne("")].copy()

    swn, _wn, lemmatizer, pos_tag = _ensure_nltk()
    vader = SentimentIntensityAnalyzer()

    vader_scores = df["text"].map(lambda x: float(vader.polarity_scores(str(x)).get("compound", 0.0)))
    swn_scores = df["text"].map(lambda x: _sentiwordnet_score(str(x), swn, lemmatizer, pos_tag))

    df["vader_compound"] = vader_scores
    df["swn_score"] = swn_scores
    df["vader_label"] = df["vader_compound"].map(lambda v: _label_from_score(float(v), args.vader_threshold))
    df["swn_label"] = df["swn_score"].map(lambda v: _label_from_score(float(v), args.swn_threshold))

    def ensemble(row: pd.Series) -> str:
        v = str(row["vader_label"])
        s = str(row["swn_label"])
        if v == s:
            return v
        return "neutral"

    df["ensemble_label"] = df.apply(ensemble, axis=1)
    df["method_agreement"] = df["vader_label"] == df["swn_label"]

    comments_path = output_dir / f"moltbook_rule_based_comments_{run_id}.csv"
    summary_path = output_dir / f"moltbook_rule_based_summary_{run_id}.json"
    label_plot_path = output_dir / f"moltbook_rule_based_label_share_{run_id}.png"
    score_plot_path = output_dir / f"moltbook_rule_based_score_distribution_{run_id}.png"

    keep_cols = [
        "comment_id",
        "post_id",
        "thread_id",
        "author_id",
        "is_verified",
        "upvotes",
        "text",
        "vader_compound",
        "swn_score",
        "vader_label",
        "swn_label",
        "ensemble_label",
        "method_agreement",
    ]
    df[keep_cols].to_csv(comments_path, index=False, encoding="utf-8")

    def share(series: pd.Series) -> Dict[str, float]:
        vals = series.value_counts(normalize=True).to_dict()
        return {str(k): round(float(v), 4) for k, v in vals.items()}

    summary = {
        "run_id": run_id,
        "input_file": str(input_path).replace("\\", "/"),
        "rows_scored": int(len(df)),
        "thresholds": {
            "vader": float(args.vader_threshold),
            "sentiwordnet": float(args.swn_threshold),
        },
        "mean_scores": {
            "vader_compound": round(float(df["vader_compound"].mean()), 4),
            "swn_score": round(float(df["swn_score"].mean()), 4),
        },
        "label_share": {
            "vader": share(df["vader_label"]),
            "sentiwordnet": share(df["swn_label"]),
            "ensemble": share(df["ensemble_label"]),
        },
        "agreement_rate": round(float(df["method_agreement"].mean()), 4),
        "primary_method": "rule_based_ensemble_vader_sentiwordnet",
        "artifacts": {
            "comments_csv": str(comments_path).replace("\\", "/"),
            "label_share_plot": str(label_plot_path).replace("\\", "/"),
            "score_distribution_plot": str(score_plot_path).replace("\\", "/"),
        },
    }

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")

    _plot_label_shares(df, label_plot_path)
    _plot_score_distributions(df, score_plot_path)

    cleanup_old_files(output_dir, "moltbook_rule_based_comments_*.csv", keep_latest=1)
    cleanup_old_files(output_dir, "moltbook_rule_based_summary_*.json", keep_latest=1)
    cleanup_old_files(output_dir, "moltbook_rule_based_label_share_*.png", keep_latest=1)
    cleanup_old_files(output_dir, "moltbook_rule_based_score_distribution_*.png", keep_latest=1)

    print("Rule-based sentiment analysis complete")
    print(f"input_file: {input_path}")
    print(f"rows_scored: {len(df)}")
    print(f"agreement_rate: {summary['agreement_rate']:.4f}")
    print(f"comments_path: {comments_path}")
    print(f"summary_path: {summary_path}")
    print(f"label_plot_path: {label_plot_path}")
    print(f"score_plot_path: {score_plot_path}")


if __name__ == "__main__":
    main()
