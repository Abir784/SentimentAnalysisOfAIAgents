from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


DATA_ROOT = Path("data")


def _latest_file(folder: Path, pattern: str) -> Path | None:
    files = sorted(folder.glob(pattern))
    if not files:
        return None
    return files[-1]


def _load_json(path: Path | None) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_dashboard_data() -> Dict[str, Any]:
    preprocessed_dir = DATA_ROOT / "preprocessed"
    polarity_dir = DATA_ROOT / "polarity"
    modeling_dir = DATA_ROOT / "modeling"
    eda_dir = DATA_ROOT / "eda"

    training_csv = _latest_file(preprocessed_dir, "moltbook_training_ready_*.csv")
    polarity_summary_json = _latest_file(polarity_dir, "moltbook_polarity_summary_*.json")
    modeling_summary_json = _latest_file(modeling_dir, "moltbook_model_summary_*.json")
    predictions_csv = _latest_file(modeling_dir, "moltbook_model_predictions_*.csv")
    eda_summary_json = _latest_file(eda_dir, "moltbook_eda_summary_*.json")

    training_df = pd.read_csv(training_csv) if training_csv and training_csv.exists() else pd.DataFrame()
    predictions_df = pd.read_csv(predictions_csv) if predictions_csv and predictions_csv.exists() else pd.DataFrame()

    return {
        "training_csv": training_csv,
        "training_df": training_df,
        "polarity_summary_json": polarity_summary_json,
        "polarity_summary": _load_json(polarity_summary_json),
        "modeling_summary_json": modeling_summary_json,
        "modeling_summary": _load_json(modeling_summary_json),
        "predictions_csv": predictions_csv,
        "predictions_df": predictions_df,
        "eda_summary_json": eda_summary_json,
        "eda_summary": _load_json(eda_summary_json),
    }


def _model_metrics_frame(modeling_summary: Dict[str, Any]) -> pd.DataFrame:
    models = modeling_summary.get("models", {})
    rows: List[Dict[str, Any]] = []
    for model_name, m in models.items():
        rows.append(
            {
                "model": model_name,
                "accuracy": float(m.get("accuracy", 0.0)),
                "f1_macro": float(m.get("f1_macro", 0.0)),
                "precision_macro": float(m.get("precision_macro", 0.0)),
                "recall_macro": float(m.get("recall_macro", 0.0)),
                "sustainability": float(m.get("sustainability", 0.0)),
                "runtime_mean_sec": float(m.get("runtime_mean_sec", 0.0)),
            }
        )
    return pd.DataFrame(rows)


def _confusion_long_frame(modeling_summary: Dict[str, Any]) -> pd.DataFrame:
    models = modeling_summary.get("models", {})
    rows: List[Dict[str, Any]] = []
    for model_name, m in models.items():
        labels = m.get("labels", [])
        cm = m.get("confusion_matrix", [])
        for i, true_label in enumerate(labels):
            for j, pred_label in enumerate(labels):
                value = 0
                if i < len(cm) and j < len(cm[i]):
                    value = int(cm[i][j])
                rows.append(
                    {
                        "model": model_name,
                        "true_label": true_label,
                        "pred_label": pred_label,
                        "count": value,
                    }
                )
    return pd.DataFrame(rows)


def _tableau_export_frames(data: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    training_df = data["training_df"]
    polarity_summary = data["polarity_summary"]
    modeling_summary = data["modeling_summary"]
    eda_summary = data["eda_summary"]

    exports: Dict[str, pd.DataFrame] = {}

    exports["model_metrics"] = _model_metrics_frame(modeling_summary)
    exports["confusion_matrix_long"] = _confusion_long_frame(modeling_summary)

    label_rows = []
    scoring = polarity_summary.get("scoring_comparison", {})
    for stage_key in ["raw_label_share", "processed_label_share"]:
        shares = scoring.get(stage_key, {})
        for label, share in shares.items():
            label_rows.append(
                {
                    "stage": stage_key.replace("_label_share", ""),
                    "label": label,
                    "share": float(share),
                }
            )
    if not label_rows:
        for label, share in polarity_summary.get("label_share", {}).items():
            label_rows.append(
                {
                    "stage": "processed",
                    "label": label,
                    "share": float(share),
                }
            )
    exports["polarity_label_share"] = pd.DataFrame(label_rows)

    rows_after_preprocess = polarity_summary.get("row_count_after_preprocessing")
    if rows_after_preprocess is None:
        rows_after_preprocess = polarity_summary.get("row_count_scored", len(training_df))

    raw_rows = polarity_summary.get("raw_row_count")
    if raw_rows is None:
        raw_rows = eda_summary.get("row_count", 0)

    dataset_rows = [
        {
            "metric": "training_rows",
            "value": float(len(training_df)),
        },
        {
            "metric": "rows_after_preprocessing",
            "value": float(rows_after_preprocess),
        },
        {
            "metric": "raw_rows",
            "value": float(raw_rows),
        },
        {
            "metric": "eda_row_count",
            "value": float(eda_summary.get("row_count", 0)),
        },
        {
            "metric": "duplicate_rows_stage",
            "value": float(eda_summary.get("duplicate_rows_by_platform_post_comment", 0)),
        },
    ]
    exports["dataset_kpis"] = pd.DataFrame(dataset_rows)

    if not training_df.empty:
        dist = (
            training_df["processed_polarity_label"]
            .value_counts(normalize=True)
            .rename_axis("label")
            .reset_index(name="share")
        )
        exports["training_label_distribution"] = dist
    else:
        exports["training_label_distribution"] = pd.DataFrame(columns=["label", "share"])

    return exports


def _author_counts_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    """Build author-level comment counts from a likely author column."""
    if df.empty:
        return pd.DataFrame(columns=["author", "comment_count"]), None

    preferred_cols = [
        "author",
        "author_name",
        "comment_author",
        "username",
        "user_name",
        "author_id",
    ]

    author_col = next((c for c in preferred_cols if c in df.columns), None)
    if author_col is None:
        fuzzy = [c for c in df.columns if "author" in c.lower() or "user" in c.lower()]
        author_col = fuzzy[0] if fuzzy else None

    if author_col is None:
        return pd.DataFrame(columns=["author", "comment_count"]), None

    authors = (
        df[author_col]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .replace("", "Unknown")
    )

    counts = (
        authors.value_counts()
        .rename_axis("author")
        .reset_index(name="comment_count")
    )
    return counts, author_col


def _first_existing(df: pd.DataFrame, columns: List[str]) -> str | None:
    """Return first column name that exists in dataframe."""
    return next((c for c in columns if c in df.columns), None)


def _fmt_metric(value: Any, decimals: int = 4) -> str:
    """Format optional numeric metric for Streamlit cards."""
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"


def main() -> None:
    st.set_page_config(
        page_title="MoltBook Sentiment Dashboard",
        page_icon="📊",
        layout="wide",
    )

    st.title("MoltBook Sentiment Analysis Dashboard")
    st.caption("Live dashboard over latest staged, polarity, EDA, and modeling artifacts.")

    data = load_dashboard_data()

    with st.sidebar:
        st.header("Data Sources")
        st.write(f"Training CSV: {data['training_csv']}")
        st.write(f"Polarity Summary: {data['polarity_summary_json']}")
        st.write(f"Model Summary: {data['modeling_summary_json']}")
        st.write(f"Predictions CSV: {data['predictions_csv']}")
        st.write(f"EDA Summary: {data['eda_summary_json']}")
        st.divider()
        st.subheader("Loaded Run IDs")
        st.write(f"Polarity Run: {data['polarity_summary'].get('run_id', 'N/A')}")
        st.write(f"Modeling Run: {data['modeling_summary'].get('run_id', 'N/A')}")
        st.write(f"EDA Run: {data['eda_summary'].get('run_id', 'N/A')}")
        st.divider()
        if st.button("Refresh cache"):
            st.cache_data.clear()
            st.rerun()

    training_df = data["training_df"]
    polarity_summary = data["polarity_summary"]
    modeling_summary = data["modeling_summary"]
    predictions_df = data["predictions_df"]
    eda_summary = data["eda_summary"]

    if training_df.empty:
        st.error("No training data found under data/preprocessed. Run polarity pipeline first.")
        return

    raw_rows = polarity_summary.get("raw_row_count")
    if raw_rows is None:
        raw_rows = eda_summary.get("row_count", 0)

    rows_after_preprocess = polarity_summary.get("row_count_after_preprocessing")
    if rows_after_preprocess is None:
        rows_after_preprocess = polarity_summary.get("row_count_scored", len(training_df))

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Training Rows", f"{len(training_df):,}")
    k2.metric("Raw Rows", f"{int(raw_rows):,}")
    k3.metric("Rows After Preprocess", f"{int(rows_after_preprocess):,}")

    model_metrics = _model_metrics_frame(modeling_summary)
    best_model = "N/A"
    best_acc = 0.0
    if not model_metrics.empty:
        idx = model_metrics["accuracy"].idxmax()
        best_model = str(model_metrics.loc[idx, "model"])
        best_acc = float(model_metrics.loc[idx, "accuracy"])
    k4.metric("Best Model", best_model, f"acc={best_acc:.4f}")

    tabs = st.tabs([
        "Overview",
        "Data Quality",
        "Polarity",
        "Modeling",
        "Predictions",
        "Tableau Export",
    ])

    with tabs[0]:
        st.subheader("Dataset Overview")
        st.caption(
            "Latest runs: "
            f"model={modeling_summary.get('run_id', 'N/A')} | "
            f"polarity={polarity_summary.get('run_id', 'N/A')} | "
            f"eda={eda_summary.get('run_id', 'N/A')}"
        )
        c1, c2 = st.columns(2)

        label_col = _first_existing(training_df, ["processed_polarity_label", "raw_polarity_label"])
        if label_col:
            label_dist = (
                training_df[label_col]
                .value_counts()
                .rename_axis("label")
                .reset_index(name="count")
            )
            fig_pie = px.pie(label_dist, values="count", names="label", title=f"Polarity Distribution ({label_col})")
            c1.plotly_chart(fig_pie, use_container_width=True)
        else:
            c1.info("No polarity label column found for overview chart.")

        if "text_len_words_traditional_clean" in training_df.columns:
            fig_hist = px.histogram(
                training_df,
                x="text_len_words_traditional_clean",
                nbins=40,
                title="Text Length Distribution (Words)",
            )
            c2.plotly_chart(fig_hist, use_container_width=True)

        st.dataframe(training_df.head(20), use_container_width=True)

    with tabs[1]:
        st.subheader("Data Quality and Preprocessing")
        drop_counts = polarity_summary.get("preprocessing", {}).get("drop_counts", {})
        if drop_counts:
            drop_df = pd.DataFrame(
                [{"step": k, "count": v} for k, v in drop_counts.items()]
            )
            fig_drop = px.bar(drop_df, x="step", y="count", title="Preprocessing Drop Counts")
            st.plotly_chart(fig_drop, use_container_width=True)
        else:
            st.info("Detailed preprocessing drop counts are not available in the latest polarity summary format.")

        if eda_summary:
            c1, c2, c3 = st.columns(3)
            c1.metric("Unique Posts", f"{eda_summary.get('unique_posts', 0):,}")
            c2.metric("Unique Authors", f"{eda_summary.get('unique_authors', 0):,}")
            c3.metric(
                "Duplicate Rows (stage)",
                f"{eda_summary.get('duplicate_rows_by_platform_post_comment', 0):,}",
            )

            missing_rate = eda_summary.get("missing_rate", {})
            if missing_rate:
                mr_df = pd.DataFrame(
                    [{"field": k, "missing_rate": float(v)} for k, v in missing_rate.items()]
                )
                fig_mr = px.bar(mr_df, x="field", y="missing_rate", title="Missing Rate by Field")
                st.plotly_chart(fig_mr, use_container_width=True)

        st.markdown("### Author Activity")
        show_author_counts = st.checkbox("Show unique authors and their comment counts", value=False)
        if show_author_counts:
            author_counts_df, author_col = _author_counts_frame(training_df)
            if author_counts_df.empty:
                st.info("No author-like column found in the training dataset.")
            else:
                top_n = st.slider("Number of authors to display", min_value=10, max_value=200, value=30, step=10)
                shown_authors = author_counts_df.head(top_n)
                fig_authors = px.bar(
                    shown_authors.sort_values("comment_count", ascending=True),
                    x="comment_count",
                    y="author",
                    orientation="h",
                    title=f"Top {top_n} Authors by Comment Count ({author_col})",
                )
                st.plotly_chart(fig_authors, use_container_width=True)
                st.dataframe(shown_authors, use_container_width=True)

    with tabs[2]:
        st.subheader("Polarity Analysis")
        scoring = polarity_summary.get("scoring_comparison", {})

        raw_mean = scoring.get("raw_mean_compound")
        processed_mean = scoring.get("processed_mean_compound")
        if processed_mean is None:
            processed_mean = polarity_summary.get("mean_compound")
        label_change_rate = scoring.get("label_change_rate")

        c1, c2, c3 = st.columns(3)
        c1.metric("Raw Mean Compound", _fmt_metric(raw_mean))
        c2.metric("Processed Mean Compound", _fmt_metric(processed_mean))
        c3.metric("Label Change Rate", _fmt_metric(label_change_rate))

        share_rows = []
        for label, val in scoring.get("raw_label_share", {}).items():
            share_rows.append({"stage": "raw", "label": label, "share": float(val)})
        for label, val in scoring.get("processed_label_share", {}).items():
            share_rows.append({"stage": "processed", "label": label, "share": float(val)})

        if not share_rows:
            for label, val in polarity_summary.get("label_share", {}).items():
                share_rows.append({"stage": "processed", "label": label, "share": float(val)})

        if share_rows:
            share_df = pd.DataFrame(share_rows)
            fig_share = px.bar(
                share_df,
                x="label",
                y="share",
                color="stage",
                barmode="group",
                title="Raw vs Processed Label Share",
            )
            st.plotly_chart(fig_share, use_container_width=True)
        else:
            st.info("No polarity share data available in current summary artifact.")

    with tabs[3]:
        st.subheader("Model Performance")
        if model_metrics.empty:
            st.warning("No modeling summary found. Run scripts/run_moltbook_modeling.py.")
        else:
            fig_acc = px.bar(
                model_metrics.sort_values("accuracy", ascending=False),
                x="model",
                y=["accuracy", "f1_macro", "precision_macro", "recall_macro"],
                barmode="group",
                title="Model Metrics",
            )
            st.plotly_chart(fig_acc, use_container_width=True)

            fig_runtime = px.scatter(
                model_metrics,
                x="runtime_mean_sec",
                y="f1_macro",
                size="accuracy",
                color="model",
                title="Runtime vs F1 (bubble size = accuracy)",
            )
            st.plotly_chart(fig_runtime, use_container_width=True)

            model_choice = st.selectbox("Confusion Matrix Model", model_metrics["model"].tolist())
            models = modeling_summary.get("models", {})
            chosen = models.get(model_choice, {})
            labels = chosen.get("labels", [])
            cm = chosen.get("confusion_matrix", [])
            if labels and cm:
                fig_cm = go.Figure(
                    data=go.Heatmap(
                        z=cm,
                        x=labels,
                        y=labels,
                        colorscale="Blues",
                        text=cm,
                        texttemplate="%{text}",
                    )
                )
                fig_cm.update_layout(title=f"Confusion Matrix: {model_choice}", xaxis_title="Predicted", yaxis_title="True")
                st.plotly_chart(fig_cm, use_container_width=True)

    with tabs[4]:
        st.subheader("Predictions Explorer")
        if predictions_df.empty:
            st.warning("No predictions CSV found.")
        else:
            pred_cols = [c for c in predictions_df.columns if c.startswith("pred_")]
            if pred_cols:
                selected_pred = st.selectbox("Prediction column", pred_cols)
                filter_label = st.multiselect(
                    "Filter predicted labels",
                    sorted(predictions_df[selected_pred].dropna().unique().tolist()),
                )
                shown = predictions_df.copy()
                if filter_label:
                    shown = shown[shown[selected_pred].isin(filter_label)]
                st.dataframe(shown.head(500), use_container_width=True)
                st.download_button(
                    label="Download filtered predictions CSV",
                    data=shown.to_csv(index=False).encode("utf-8"),
                    file_name="filtered_predictions.csv",
                    mime="text/csv",
                )
            else:
                st.dataframe(predictions_df.head(500), use_container_width=True)

    with tabs[5]:
        st.subheader("Tableau-Ready Exports")
        st.caption("Use these extracted tables directly in Tableau Desktop/Public.")

        exports = _tableau_export_frames(data)
        export_name = st.selectbox("Export dataset", list(exports.keys()))
        export_df = exports[export_name]
        st.dataframe(export_df.head(200), use_container_width=True)
        st.download_button(
            label=f"Download {export_name}.csv",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{export_name}.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
