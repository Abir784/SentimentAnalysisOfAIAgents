from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
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
    raw_dir = DATA_ROOT / "raw"
    staged_dir = DATA_ROOT / "staged"
    pre_dir = DATA_ROOT / "preprocessed_rule_based"
    eda_dir = DATA_ROOT / "eda_rule_based"
    feat_dir = DATA_ROOT / "features_rule_based"
    rule_dir = DATA_ROOT / "rule_based"
    interaction_dir = DATA_ROOT / "eda"

    raw_latest = _latest_file(raw_dir, "moltbook_raw_*.jsonl")
    staged_latest = staged_dir / "moltbook_comments_all.jsonl"
    pre_latest = _latest_file(pre_dir, "moltbook_preprocessed_rule_based_*.csv")
    eda_latest = _latest_file(eda_dir, "moltbook_eda_rule_based_summary_*.json")
    feat_latest = _latest_file(feat_dir, "moltbook_features_rule_based_*.csv")
    feat_summary_latest = _latest_file(feat_dir, "moltbook_features_rule_based_summary_*.json")
    rule_summary_latest = _latest_file(rule_dir, "moltbook_rule_based_summary_*.json")
    rule_comments_latest = _latest_file(rule_dir, "moltbook_rule_based_comments_*.csv")
    rule_label_plot = _latest_file(rule_dir, "moltbook_rule_based_label_share_*.png")
    rule_score_plot = _latest_file(rule_dir, "moltbook_rule_based_score_distribution_*.png")

    interaction_summary = _latest_file(interaction_dir, "moltbook_interaction_network_summary_*.json")
    interaction_nodes = _latest_file(interaction_dir, "moltbook_interaction_network_nodes_*.csv")
    interaction_threads = _latest_file(interaction_dir, "moltbook_interaction_network_thread_stats_*.csv")
    interaction_topology = interaction_dir / "moltbook_interaction_network_topology_latest.png"
    interaction_dist = interaction_dir / "moltbook_interaction_network_distributions_latest.png"

    return {
        "raw_latest": raw_latest,
        "staged_latest": staged_latest if staged_latest.exists() else None,
        "pre_latest": pre_latest,
        "eda_summary": _load_json(eda_latest),
        "feature_summary": _load_json(feat_summary_latest),
        "feature_df": pd.read_csv(feat_latest) if feat_latest and feat_latest.exists() else pd.DataFrame(),
        "rule_summary": _load_json(rule_summary_latest),
        "rule_df": pd.read_csv(rule_comments_latest) if rule_comments_latest and rule_comments_latest.exists() else pd.DataFrame(),
        "rule_label_plot": rule_label_plot,
        "rule_score_plot": rule_score_plot,
        "interaction_summary": _load_json(interaction_summary),
        "interaction_nodes_df": pd.read_csv(interaction_nodes) if interaction_nodes and interaction_nodes.exists() else pd.DataFrame(),
        "interaction_thread_df": pd.read_csv(interaction_threads) if interaction_threads and interaction_threads.exists() else pd.DataFrame(),
        "interaction_topology": interaction_topology if interaction_topology.exists() else None,
        "interaction_dist": interaction_dist if interaction_dist.exists() else None,
        "paths": {
            "raw": raw_latest,
            "staged": staged_latest if staged_latest.exists() else None,
            "preprocessed": pre_latest,
            "eda": eda_latest,
            "features": feat_latest,
            "rule_summary": rule_summary_latest,
            "rule_comments": rule_comments_latest,
            "interaction_summary": interaction_summary,
        },
    }


def _fmt(value: Any, dec: int = 4) -> str:
    try:
        return f"{float(value):.{dec}f}"
    except Exception:
        return "N/A"


def main() -> None:
    st.set_page_config(page_title="MoltBook Rule-Based Dashboard", page_icon="📊", layout="wide")
    st.title("MoltBook Sentiment Analysis Dashboard")
    st.caption("Rule-based pipeline dashboard (VADER + SentiWordNet + Ensemble) with separate RQ1 analysis")

    data = load_dashboard_data()

    with st.sidebar:
        st.header("Pipeline Sources")
        for k, v in data["paths"].items():
            st.write(f"{k}: {v}")
        if st.button("Refresh cache"):
            st.cache_data.clear()
            st.rerun()

    rule_summary = data["rule_summary"]
    interaction_summary = data["interaction_summary"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows Scored", f"{int(rule_summary.get('rows_scored', 0)):,}")
    c2.metric("Agreement Rate", _fmt(rule_summary.get("agreement_rate"), 4))
    c3.metric("RQ1 Nodes", f"{int(interaction_summary.get('node_count', 0)):,}")
    c4.metric("RQ1 Edges", f"{int(interaction_summary.get('edge_count', 0)):,}")

    tabs = st.tabs(["Overview", "Rule-Based Results", "Feature Extraction", "RQ1 Analysis"])

    with tabs[0]:
        st.subheader("Pipeline Overview")
        st.markdown(
            """
1. Data Acquisition
2. Text Preprocessing
3. EDA
4. Feature Extraction
5. 3 Rule-Based Analysis Tools (VADER, SentiWordNet, Ensemble)

RQ1 Interaction Network is run separately.
"""
        )

        label_share = rule_summary.get("label_share", {})
        rows: List[Dict[str, Any]] = []
        for method, mapping in label_share.items():
            for label, share in (mapping or {}).items():
                rows.append({"method": str(method), "label": str(label), "share": float(share)})
        if rows:
            share_df = pd.DataFrame(rows)
            fig = px.bar(share_df, x="label", y="share", color="method", barmode="group", title="Rule-Based Label Share")
            st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        st.subheader("Rule-Based Sentiment Results")
        st.json(rule_summary)

        label_plot = data.get("rule_label_plot")
        score_plot = data.get("rule_score_plot")
        if label_plot and Path(label_plot).exists():
            st.image(str(label_plot), caption="Rule-Based Label Share Snapshot", use_container_width=True)
        if score_plot and Path(score_plot).exists():
            st.image(str(score_plot), caption="Rule-Based Score Distribution Snapshot", use_container_width=True)

        if not data["rule_df"].empty:
            st.dataframe(data["rule_df"].head(200), use_container_width=True)

    with tabs[2]:
        st.subheader("Feature Extraction")
        st.json(data.get("feature_summary", {}))
        feat_df = data.get("feature_df", pd.DataFrame())
        if feat_df.empty:
            st.info("No feature extraction output found yet.")
        else:
            st.dataframe(feat_df.head(200), use_container_width=True)
            numeric_cols = [c for c in feat_df.columns if c not in {"comment_id", "post_id", "thread_id", "author_id"}]
            if numeric_cols:
                sel = st.selectbox("Feature distribution", numeric_cols)
                fig = px.histogram(feat_df, x=sel, nbins=40, title=f"Distribution: {sel}")
                st.plotly_chart(fig, use_container_width=True)

    with tabs[3]:
        st.subheader("RQ1 Interaction Network (Separate)")
        st.caption(
            f"Mode: {interaction_summary.get('edge_construction_mode', 'N/A')} | "
            f"Fallback: {interaction_summary.get('fallback_triggered', 'N/A')}"
        )

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Nodes", f"{int(interaction_summary.get('node_count', 0)):,}")
        d2.metric("Edges", f"{int(interaction_summary.get('edge_count', 0)):,}")
        d3.metric("Reciprocity", _fmt(interaction_summary.get("reciprocity"), 3))
        d4.metric("Clustering", _fmt(interaction_summary.get("average_undirected_clustering"), 3))

        topo = data.get("interaction_topology")
        dist = data.get("interaction_dist")
        if topo:
            st.image(str(topo), caption="Interaction Topology", use_container_width=True)
        if dist:
            st.image(str(dist), caption="Interaction Distributions", use_container_width=True)

        nodes_df = data.get("interaction_nodes_df", pd.DataFrame())
        thread_df = data.get("interaction_thread_df", pd.DataFrame())
        if not nodes_df.empty:
            st.markdown("Top interaction nodes")
            st.dataframe(nodes_df.head(30), use_container_width=True)
        if not thread_df.empty:
            fig = px.histogram(thread_df, x="reply_edges", nbins=30, title="Reply Edges per Thread")
            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
