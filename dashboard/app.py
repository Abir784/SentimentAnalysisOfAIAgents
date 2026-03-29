from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


DATA_ROOT = Path("data")
MODELING_RESULT_LOG = DATA_ROOT / "modeling" / "result.txt"


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


def _parse_result_log(log_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Parse modeling result.txt into run-level and metric-level dataframes."""
    if not log_path.exists():
        empty_runs = pd.DataFrame(
            columns=[
                "run_id",
                "saved_at_utc",
                "input_csv",
                "rows_used",
                "cv_folds",
                "best_accuracy_model",
                "best_accuracy",
                "best_f1_model",
                "best_f1_macro",
            ]
        )
        empty_metrics = pd.DataFrame(
            columns=["run_id", "model", "accuracy", "f1_macro", "precision_macro", "recall_macro", "sustainability"]
        )
        return empty_runs, empty_metrics, ""

    raw = log_path.read_text(encoding="utf-8")
    chunks = [c.strip() for c in re.split(r"\n={20,}\n", raw) if c.strip()]

    runs_rows: List[Dict[str, Any]] = []
    metric_rows: List[Dict[str, Any]] = []

    for chunk in chunks:
        run_match = re.search(r"^Run ID:\s*(\S+)", chunk, flags=re.MULTILINE)
        if not run_match:
            continue
        run_id = run_match.group(1)

        saved_at = re.search(r"^Saved At \(UTC\):\s*(.+)$", chunk, flags=re.MULTILINE)
        input_csv = re.search(r"^Input CSV:\s*(.+)$", chunk, flags=re.MULTILINE)
        rows_used = re.search(r"^Rows Used:\s*(\d+)$", chunk, flags=re.MULTILINE)
        cv_folds = re.search(r"^CV Folds:\s*(\d+)$", chunk, flags=re.MULTILINE)
        best_acc = re.search(r"^Best Accuracy:\s*(\S+)\s*\(([-+]?[0-9]*\.?[0-9]+)\)$", chunk, flags=re.MULTILINE)
        best_f1 = re.search(r"^Best Macro F1:\s*(\S+)\s*\(([-+]?[0-9]*\.?[0-9]+)\)$", chunk, flags=re.MULTILINE)

        runs_rows.append(
            {
                "run_id": run_id,
                "saved_at_utc": saved_at.group(1).strip() if saved_at else "",
                "input_csv": input_csv.group(1).strip() if input_csv else "",
                "rows_used": int(rows_used.group(1)) if rows_used else 0,
                "cv_folds": int(cv_folds.group(1)) if cv_folds else 0,
                "best_accuracy_model": best_acc.group(1) if best_acc else "",
                "best_accuracy": float(best_acc.group(2)) if best_acc else 0.0,
                "best_f1_model": best_f1.group(1) if best_f1 else "",
                "best_f1_macro": float(best_f1.group(2)) if best_f1 else 0.0,
            }
        )

        for line in chunk.splitlines():
            line = line.strip()
            if not line.startswith("- "):
                continue
            if "| accuracy=" not in line:
                continue

            model_name = line[2:].split("|")[0].strip()

            def _extract(name: str) -> float:
                m = re.search(rf"{name}=([-+]?[0-9]*\.?[0-9]+)", line)
                return float(m.group(1)) if m else 0.0

            metric_rows.append(
                {
                    "run_id": run_id,
                    "model": model_name,
                    "accuracy": _extract("accuracy"),
                    "f1_macro": _extract("f1_macro"),
                    "precision_macro": _extract("precision_macro"),
                    "recall_macro": _extract("recall_macro"),
                    "sustainability": _extract("sustainability"),
                }
            )

    runs_df = pd.DataFrame(runs_rows)
    metrics_df = pd.DataFrame(metric_rows)
    if not runs_df.empty:
        runs_df = runs_df.sort_values("saved_at_utc", ascending=False).reset_index(drop=True)
    return runs_df, metrics_df, raw


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
        "Run History",
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
            st.caption(
                "Duplicate rows in staged data are handled during preprocessing "
                "(duplicate comments are removed before training and polarity outputs are produced)."
            )

            missing_rate = eda_summary.get("missing_rate", {})
            if missing_rate:
                mr_df = pd.DataFrame(
                    [{"field": k, "missing_rate": float(v)} for k, v in missing_rate.items()]
                )
                fig_mr = px.bar(mr_df, x="field", y="missing_rate", title="Missing Rate by Field")
                st.plotly_chart(fig_mr, use_container_width=True)

        st.markdown("### Author Activity")
        top_authors_stage = eda_summary.get("top_authors_by_comment_count", [])
        if top_authors_stage:
            top_stage_df = pd.DataFrame(top_authors_stage, columns=["author", "comment_count"])
            fig_stage_authors = px.bar(
                top_stage_df.sort_values("comment_count", ascending=True),
                x="comment_count",
                y="author",
                orientation="h",
                title="Top Authors by Comment Count (Staged Data)",
            )
            st.plotly_chart(fig_stage_authors, use_container_width=True)
            st.dataframe(top_stage_df, use_container_width=True)
        else:
            st.info("Top-author summary is not available in the EDA artifact.")

        show_preprocessed_author_counts = st.checkbox(
            "Also show author counts after preprocessing",
            value=False,
        )
        if show_preprocessed_author_counts:
            author_counts_df, author_col = _author_counts_frame(training_df)
            if author_counts_df.empty:
                st.info("No author-like column found in the training dataset.")
            else:
                top_n = st.slider("Number of preprocessed authors to display", min_value=10, max_value=200, value=30, step=10)
                shown_authors = author_counts_df.head(top_n)
                fig_authors = px.bar(
                    shown_authors.sort_values("comment_count", ascending=True),
                    x="comment_count",
                    y="author",
                    orientation="h",
                    title=f"Top {top_n} Authors after Preprocessing ({author_col})",
                )
                st.plotly_chart(fig_authors, use_container_width=True)
                st.dataframe(shown_authors, use_container_width=True)

    with tabs[2]:
        st.subheader("Polarity Analysis")
        scoring = polarity_summary.get("scoring_comparison", {})

        c1, c2, c3 = st.columns(3)
        if scoring:
            raw_mean = scoring.get("raw_mean_compound")
            processed_mean = scoring.get("processed_mean_compound")
            label_change_rate = scoring.get("label_change_rate")

            c1.metric("Raw Mean Compound", _fmt_metric(raw_mean))
            c2.metric("Processed Mean Compound", _fmt_metric(processed_mean))
            c3.metric("Label Change Rate", _fmt_metric(label_change_rate))
        else:
            processed_mean = polarity_summary.get("mean_compound")
            rows_scored = polarity_summary.get("row_count_scored", len(training_df))
            compact_shares = polarity_summary.get("label_share", {})

            top_label = "N/A"
            top_share = None
            if compact_shares:
                top_label, top_share = max(compact_shares.items(), key=lambda kv: float(kv[1]))

            c1.metric("Processed Mean Compound", _fmt_metric(processed_mean))
            c2.metric("Rows Scored", f"{int(rows_scored):,}")
            c3.metric("Top Label Share", _fmt_metric(top_share), delta=f"label={top_label}")

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

    with tabs[6]:
        st.subheader("Model Run History")
        st.caption("Historical run log parsed from data/modeling/result.txt")

        runs_df, run_metrics_df, run_log_raw = _parse_result_log(MODELING_RESULT_LOG)

        if runs_df.empty:
            st.warning("No run history found in data/modeling/result.txt")
        else:
            s1, s2, s3 = st.columns(3)
            s1.metric("Total Runs", f"{len(runs_df):,}")
            s2.metric("Latest Run", str(runs_df.iloc[0]["run_id"]))
            s3.metric("Best Accuracy (All Runs)", f"{runs_df['best_accuracy'].max():.4f}")

            st.markdown("### Runs Table")
            st.dataframe(runs_df, use_container_width=True)

            selected_run = st.selectbox("Inspect Run", runs_df["run_id"].tolist())
            selected_metrics = run_metrics_df[run_metrics_df["run_id"] == selected_run].copy()

            if selected_metrics.empty:
                st.info("No model metric lines found for selected run.")
            else:
                st.markdown("### Per-Model Metrics")
                st.dataframe(
                    selected_metrics.sort_values("accuracy", ascending=False),
                    use_container_width=True,
                )

                fig_hist = px.bar(
                    selected_metrics.sort_values("accuracy", ascending=False),
                    x="model",
                    y=["accuracy", "f1_macro", "precision_macro", "recall_macro"],
                    barmode="group",
                    title=f"Run {selected_run}: Model Metric Comparison",
                )
                st.plotly_chart(fig_hist, use_container_width=True)

        with st.expander("Show Raw result.txt"):
            st.text(run_log_raw if run_log_raw else "(empty)")


if __name__ == "__main__":
    main()
