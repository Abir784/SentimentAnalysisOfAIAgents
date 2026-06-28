from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "did",
    "do",
    "does",
    "doing",
    "don",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
}


def load_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    malformed = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue

            sentiment = item.get("sentiment") or {}
            rows.append(
                {
                    "line_number": line_number,
                    "timestamp": item.get("timestamp"),
                    "model": item.get("model", "unknown"),
                    "message": item.get("message", ""),
                    "sentiment_label": item.get("sentiment_label", "unknown"),
                    "positive": sentiment.get("positive"),
                    "negative": sentiment.get("negative"),
                    "neutral": sentiment.get("neutral"),
                    "compound": sentiment.get("compound"),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No usable JSONL records found in {path}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for column in ["positive", "negative", "neutral", "compound"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["message_length"] = df["message"].fillna("").str.len()
    df.attrs["malformed_rows"] = malformed
    return df


def tokenize(messages: pd.Series) -> list[str]:
    words: list[str] = []
    for message in messages.dropna():
        for word in re.findall(r"[a-zA-Z][a-zA-Z'-]{2,}", message.lower()):
            if word not in STOPWORDS:
                words.append(word)
    return words


def top_terms(df: pd.DataFrame, limit: int = 20) -> pd.DataFrame:
    rows = []
    for label, group in df.groupby("sentiment_label"):
        for term, count in Counter(tokenize(group["message"])).most_common(limit):
            rows.append({"sentiment_label": label, "term": term, "count": count})
    return pd.DataFrame(rows)


def build_summary(df: pd.DataFrame) -> dict:
    model_summary = (
        df.groupby("model")
        .agg(
            records=("model", "size"),
            avg_compound=("compound", "mean"),
            median_compound=("compound", "median"),
            avg_positive=("positive", "mean"),
            avg_negative=("negative", "mean"),
            avg_neutral=("neutral", "mean"),
            avg_message_length=("message_length", "mean"),
        )
        .round(4)
        .sort_values("avg_compound", ascending=False)
    )

    label_counts = (
        df.groupby(["model", "sentiment_label"]).size().rename("count").reset_index()
    )
    totals = label_counts.groupby("model")["count"].transform("sum")
    label_counts["share"] = (label_counts["count"] / totals).round(4)

    return {
        "records": int(len(df)),
        "malformed_rows": int(df.attrs.get("malformed_rows", 0)),
        "models": model_summary.reset_index().to_dict(orient="records"),
        "labels": label_counts.to_dict(orient="records"),
        "overall_labels": df["sentiment_label"].value_counts().to_dict(),
    }


def create_report(df: pd.DataFrame, input_path: Path, output_html: Path) -> None:
    px.defaults.template = "plotly_white"
    palette = {"negative": "#d04f45", "neutral": "#8a8f98", "positive": "#16875d"}

    label_counts = (
        df.groupby(["model", "sentiment_label"]).size().rename("count").reset_index()
    )
    label_counts["share"] = label_counts["count"] / label_counts.groupby("model")[
        "count"
    ].transform("sum")

    model_summary = (
        df.groupby("model")
        .agg(
            records=("model", "size"),
            avg_compound=("compound", "mean"),
            avg_positive=("positive", "mean"),
            avg_negative=("negative", "mean"),
            avg_neutral=("neutral", "mean"),
            avg_message_length=("message_length", "mean"),
        )
        .reset_index()
        .sort_values("avg_compound", ascending=False)
    )

    labels_order = ["negative", "neutral", "positive"]
    fig_labels = px.bar(
        label_counts,
        x="model",
        y="share",
        color="sentiment_label",
        category_orders={"sentiment_label": labels_order},
        color_discrete_map=palette,
        text=label_counts["share"].map(lambda value: f"{value:.0%}"),
        title="Sentiment Label Share by Model",
    )
    fig_labels.update_layout(yaxis_tickformat=".0%", yaxis_title="Share of outputs")
    fig_labels.update_traces(textposition="inside")

    fig_compound = px.bar(
        model_summary,
        x="model",
        y="avg_compound",
        color="avg_compound",
        color_continuous_scale=["#d04f45", "#f4d35e", "#16875d"],
        title="Average Compound Sentiment by Model",
        text=model_summary["avg_compound"].map(lambda value: f"{value:.3f}"),
    )
    fig_compound.add_hline(y=0, line_dash="dash", line_color="#4d5562")
    fig_compound.update_layout(coloraxis_showscale=False, yaxis_title="Avg compound")
    fig_compound.update_traces(textposition="outside", cliponaxis=False)

    fig_distribution = px.violin(
        df,
        x="model",
        y="compound",
        color="model",
        box=True,
        points=False,
        title="Compound Sentiment Distribution",
    )
    fig_distribution.add_hline(y=0, line_dash="dash", line_color="#4d5562")
    fig_distribution.update_layout(showlegend=False)

    components = model_summary.melt(
        id_vars="model",
        value_vars=["avg_positive", "avg_negative", "avg_neutral"],
        var_name="component",
        value_name="score",
    )
    components["component"] = components["component"].str.replace("avg_", "", regex=False)
    fig_components = px.bar(
        components,
        x="model",
        y="score",
        color="component",
        barmode="group",
        color_discrete_map={
            "positive": "#16875d",
            "negative": "#d04f45",
            "neutral": "#6d7786",
        },
        title="Average VADER Component Scores",
    )

    df_ordered = df.sort_values(["timestamp", "line_number"]).copy()
    df_ordered["sequence"] = range(1, len(df_ordered) + 1)
    df_ordered["rolling_compound"] = (
        df_ordered.groupby("model")["compound"]
        .transform(lambda values: values.rolling(50, min_periods=5).mean())
    )
    fig_timeline = px.line(
        df_ordered,
        x="sequence",
        y="rolling_compound",
        color="model",
        title="Rolling Compound Sentiment Across Generation Order",
    )
    fig_timeline.add_hline(y=0, line_dash="dash", line_color="#4d5562")
    fig_timeline.update_layout(yaxis_title="Rolling avg compound", xaxis_title="Record order")

    length_summary = (
        df.groupby("model")["message_length"]
        .agg(["mean", "min", "max"])
        .round(1)
        .reset_index()
    )
    fig_length = px.bar(
        length_summary,
        x="model",
        y="mean",
        color="model",
        text=length_summary["mean"].map(lambda value: f"{value:.1f}"),
        title="Average Message Length",
    )
    fig_length.update_layout(showlegend=False, yaxis_title="Characters")
    fig_length.update_traces(textposition="outside", cliponaxis=False)

    terms = top_terms(df, limit=12)
    fig_terms = px.bar(
        terms,
        x="count",
        y="term",
        color="sentiment_label",
        facet_col="sentiment_label",
        facet_col_wrap=3,
        color_discrete_map=palette,
        title="Most Common Message Terms by Sentiment Label",
    )
    fig_terms.update_yaxes(matches=None, showticklabels=True)
    fig_terms.update_layout(showlegend=False)

    table = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=[
                        "Model",
                        "Records",
                        "Avg compound",
                        "Avg positive",
                        "Avg negative",
                        "Avg neutral",
                        "Avg length",
                    ],
                    fill_color="#1f2937",
                    font=dict(color="white"),
                    align="left",
                ),
                cells=dict(
                    values=[
                        model_summary["model"],
                        model_summary["records"],
                        model_summary["avg_compound"].round(3),
                        model_summary["avg_positive"].round(3),
                        model_summary["avg_negative"].round(3),
                        model_summary["avg_neutral"].round(3),
                        model_summary["avg_message_length"].round(1),
                    ],
                    align="left",
                ),
            )
        ]
    )
    table.update_layout(title="Model Summary Table")

    figures = [
        fig_labels,
        fig_compound,
        fig_distribution,
        fig_components,
        fig_timeline,
        fig_length,
        fig_terms,
        table,
    ]

    summary = build_summary(df)
    best_model = max(summary["models"], key=lambda row: row["avg_compound"])
    lowest_model = min(summary["models"], key=lambda row: row["avg_compound"])
    total = summary["records"]
    overall = summary["overall_labels"]

    charts_html = "\n".join(
        fig.to_html(full_html=False, include_plotlyjs=(index == 0))
        for index, fig in enumerate(figures)
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Synthetic Sentiment Analysis Visual Report</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #17202a;
      --muted: #5f6b7a;
      --line: #dbe1e8;
      --panel: #ffffff;
      --bg: #f5f7fa;
      --accent: #1768ac;
    }}
    body {{
      margin: 0;
      font-family: "Segoe UI", Arial, sans-serif;
      color: var(--ink);
      background: var(--bg);
    }}
    header {{
      background: #ffffff;
      border-bottom: 1px solid var(--line);
      padding: 28px min(6vw, 64px);
    }}
    main {{
      padding: 24px min(6vw, 64px) 48px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 30px;
      letter-spacing: 0;
    }}
    .subtle {{
      color: var(--muted);
      margin: 0;
      line-height: 1.5;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin: 20px 0;
    }}
    .metric {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }}
    .metric strong {{
      display: block;
      font-size: 24px;
      margin-bottom: 4px;
    }}
    .metric span {{
      color: var(--muted);
      font-size: 13px;
    }}
    .insights {{
      background: #eef6ff;
      border: 1px solid #c5dcf5;
      border-radius: 8px;
      padding: 16px 18px;
      margin-bottom: 20px;
    }}
    .insights h2 {{
      margin: 0 0 8px;
      font-size: 18px;
    }}
    .insights ul {{
      margin: 0;
      padding-left: 20px;
      line-height: 1.6;
    }}
    .chart {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      margin: 16px 0;
      padding: 8px;
    }}
    .js-plotly-plot {{
      width: 100%;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Synthetic Sentiment Analysis Visual Report</h1>
    <p class="subtle">Source: {input_path.as_posix()} | Generated from {total:,} valid rows.</p>
  </header>
  <main>
    <section class="metrics">
      <div class="metric"><strong>{total:,}</strong><span>Total analyzed rows</span></div>
      <div class="metric"><strong>{len(summary["models"])}</strong><span>Models compared</span></div>
      <div class="metric"><strong>{overall.get("negative", 0):,}</strong><span>Negative outputs</span></div>
      <div class="metric"><strong>{overall.get("positive", 0):,}</strong><span>Positive outputs</span></div>
      <div class="metric"><strong>{overall.get("neutral", 0):,}</strong><span>Neutral outputs</span></div>
    </section>
    <section class="insights">
      <h2>Highlights</h2>
      <ul>
        <li><strong>{best_model["model"]}</strong> has the highest average compound score ({best_model["avg_compound"]:.3f}).</li>
        <li><strong>{lowest_model["model"]}</strong> has the lowest average compound score ({lowest_model["avg_compound"]:.3f}).</li>
        <li>The overall label mix is {overall.get("negative", 0):,} negative, {overall.get("positive", 0):,} positive, and {overall.get("neutral", 0):,} neutral.</li>
      </ul>
    </section>
    <section class="chart">
      {charts_html}
    </section>
  </main>
</body>
</html>
"""
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create visual summaries for synthetic sentiment JSONL results."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/sythetic/analysis_results.jsonl"),
        help="Path to JSONL sentiment analysis results.",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=Path("data/sythetic/sentiment_visual_report.html"),
        help="Self-contained HTML report path.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("data/sythetic/sentiment_summary.json"),
        help="Machine-readable summary output path.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("data/sythetic/model_sentiment_summary.csv"),
        help="Model summary CSV output path.",
    )
    args = parser.parse_args()

    df = load_jsonl(args.input)
    create_report(df, args.input, args.output_html)

    summary = build_summary(df)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(summary["models"]).to_csv(args.summary_csv, index=False)

    print(f"Analyzed {summary['records']:,} rows")
    print(f"HTML report: {args.output_html}")
    print(f"Summary JSON: {args.summary_json}")
    print(f"Summary CSV: {args.summary_csv}")


if __name__ == "__main__":
    main()
