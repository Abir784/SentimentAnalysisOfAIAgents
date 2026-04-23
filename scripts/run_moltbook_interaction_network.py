from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.file_management import cleanup_old_files

try:
    import networkx as nx
except ImportError as exc:
    raise ImportError(
        "networkx is required for interaction-network analysis. "
        "Install dependencies with: pip install -r requirements.txt"
    ) from exc


EDGE_MODE_JUSTIFICATION = """Edge construction modes for MoltBook interaction graphs:

1) direct mode
     - Requires explicit parent-comment linkage where each comment's parent_id matches
         another comment_id in the same staged corpus.
     - On this dataset, direct mode failed to produce usable reply edges because parent_id
         values frequently do not resolve to in-corpus comment_id records at analysis time.

2) sequential mode
     - Constructs reply edges from consecutive comments in the same thread, preserving
         observed thread order as a proxy reply chain: author(i-1) -> author(i).
     - Assumes adjacency is a reasonable interaction signal when explicit reply metadata
         is missing or unresolved.

3) why sequential fallback is acceptable here
     - MoltBook staged data consistently preserves post/thread grouping and comment order.
     - Sequential fallback keeps all thread interaction signal available for descriptive
         structure analysis while clearly labeling results as fallback-derived.
     - This supports exploratory network analysis without overstating direct-reply certainty.
"""


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _to_author(value: Any) -> str:
    text = str(value or "").strip()
    return text if text else "unknown_author"


def _build_direct_edge_table(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    id_to_author: Dict[str, str] = {}
    for row in df.itertuples(index=False):
        comment_id = str(getattr(row, "comment_id", "") or "").strip()
        if not comment_id:
            continue
        id_to_author[comment_id] = _to_author(getattr(row, "author_id", ""))

    edge_counts: Counter[Tuple[str, str]] = Counter()
    thread_edge_counts: Counter[Tuple[str, str, str]] = Counter()

    counts = {
        "rows_total": int(len(df)),
        "rows_with_comment_id": int(df["comment_id"].astype(str).str.strip().ne("").sum()),
        "rows_with_parent_id": int(df["parent_id"].astype(str).str.strip().ne("").sum()),
        "rows_parent_resolved_to_comment": 0,
        "rows_parent_not_comment": 0,
        "self_reply_rows_skipped": 0,
    }

    for row in df.itertuples(index=False):
        child_author = _to_author(getattr(row, "author_id", ""))
        parent_id = str(getattr(row, "parent_id", "") or "").strip()
        thread_id = str(getattr(row, "thread_id", "") or "").strip()

        if not parent_id:
            counts["rows_parent_not_comment"] += 1
            continue

        parent_author = id_to_author.get(parent_id)
        if parent_author is None:
            counts["rows_parent_not_comment"] += 1
            continue

        counts["rows_parent_resolved_to_comment"] += 1

        if parent_author == child_author:
            counts["self_reply_rows_skipped"] += 1
            continue

        edge_counts[(parent_author, child_author)] += 1
        thread_edge_counts[(thread_id, parent_author, child_author)] += 1

    edge_rows = [
        {"source_author": src, "target_author": dst, "weight": int(weight)}
        for (src, dst), weight in edge_counts.items()
    ]
    edge_df = pd.DataFrame(edge_rows)

    if edge_df.empty:
        edge_df = pd.DataFrame(columns=["source_author", "target_author", "weight"])

    thread_edge_rows = [
        {
            "thread_id": tid,
            "source_author": src,
            "target_author": dst,
            "weight": int(weight),
        }
        for (tid, src, dst), weight in thread_edge_counts.items()
    ]
    thread_edge_df = pd.DataFrame(thread_edge_rows)
    if thread_edge_df.empty:
        thread_edge_df = pd.DataFrame(columns=["thread_id", "source_author", "target_author", "weight"])

    counts["unique_reply_edges"] = int(len(edge_df))
    counts["total_reply_interactions"] = int(edge_df["weight"].sum()) if not edge_df.empty else 0
    counts["edge_mode"] = "direct_parent_reply"

    return (edge_df, thread_edge_df, counts)


def _build_sequential_edge_table(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    work = df.copy()
    work["thread_id"] = work["thread_id"].fillna("").astype(str)
    work["author_id"] = work["author_id"].fillna("").astype(str)

    edge_counts: Counter[Tuple[str, str]] = Counter()
    thread_edge_counts: Counter[Tuple[str, str, str]] = Counter()

    counts = {
        "rows_total": int(len(work)),
        "rows_with_thread_id": int(work["thread_id"].astype(str).str.strip().ne("").sum()),
        "threads_with_2plus_comments": 0,
        "sequential_pairs_considered": 0,
        "self_reply_rows_skipped": 0,
        "rows_missing_thread_id": 0,
    }

    for thread_id, group in work.groupby("thread_id", sort=False, dropna=False):
        tid = str(thread_id or "").strip()
        if not tid:
            counts["rows_missing_thread_id"] += int(len(group))
            continue

        if len(group) < 2:
            continue
        counts["threads_with_2plus_comments"] += 1

        # Preserve original dataset order inside each thread as interaction sequence.
        ordered = group.copy()
        authors = [_to_author(a) for a in ordered["author_id"].tolist()]

        for i in range(1, len(authors)):
            prev_author = authors[i - 1]
            curr_author = authors[i]
            counts["sequential_pairs_considered"] += 1

            if prev_author == curr_author:
                counts["self_reply_rows_skipped"] += 1
                continue

            edge_counts[(prev_author, curr_author)] += 1
            thread_edge_counts[(tid, prev_author, curr_author)] += 1

    edge_rows = [
        {"source_author": src, "target_author": dst, "weight": int(weight)}
        for (src, dst), weight in edge_counts.items()
    ]
    edge_df = pd.DataFrame(edge_rows)
    if edge_df.empty:
        edge_df = pd.DataFrame(columns=["source_author", "target_author", "weight"])

    thread_edge_rows = [
        {
            "thread_id": tid,
            "source_author": src,
            "target_author": dst,
            "weight": int(weight),
        }
        for (tid, src, dst), weight in thread_edge_counts.items()
    ]
    thread_edge_df = pd.DataFrame(thread_edge_rows)
    if thread_edge_df.empty:
        thread_edge_df = pd.DataFrame(columns=["thread_id", "source_author", "target_author", "weight"])

    counts["unique_reply_edges"] = int(len(edge_df))
    counts["total_reply_interactions"] = int(edge_df["weight"].sum()) if not edge_df.empty else 0
    counts["edge_mode"] = "sequential_thread_fallback"

    return (edge_df, thread_edge_df, counts)


def _select_edge_tables(
    df: pd.DataFrame,
    edge_mode: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int], Dict[str, Any]]:
    mode = edge_mode.strip().lower()
    if mode not in {"auto", "direct", "sequential"}:
        raise ValueError("--edge-mode must be one of: auto, direct, sequential")

    direct_edge_df, direct_thread_edge_df, direct_counts = _build_direct_edge_table(df)
    sequential_edge_df, sequential_thread_edge_df, sequential_counts = _build_sequential_edge_table(df)

    diagnostics: Dict[str, Any] = {
        "direct_edge_count": int(len(direct_edge_df)),
        "direct_weighted_interactions": int(direct_edge_df["weight"].sum()) if not direct_edge_df.empty else 0,
        "sequential_edge_count": int(len(sequential_edge_df)),
        "sequential_weighted_interactions": int(sequential_edge_df["weight"].sum())
        if not sequential_edge_df.empty
        else 0,
        "fallback_triggered": False,
    }

    if mode == "direct":
        diagnostics["selected_mode"] = "direct"
        return direct_edge_df, direct_thread_edge_df, direct_counts, diagnostics

    if mode == "sequential":
        diagnostics["selected_mode"] = "sequential"
        return sequential_edge_df, sequential_thread_edge_df, sequential_counts, diagnostics

    # auto mode: prefer explicit parent-child edges, fallback to sequential when empty
    if not direct_edge_df.empty:
        diagnostics["selected_mode"] = "direct"
        return direct_edge_df, direct_thread_edge_df, direct_counts, diagnostics

    diagnostics["selected_mode"] = "sequential"
    diagnostics["fallback_triggered"] = True
    return sequential_edge_df, sequential_thread_edge_df, sequential_counts, diagnostics


def _build_graph(edge_df: pd.DataFrame) -> nx.DiGraph:
    graph = nx.DiGraph()
    if edge_df.empty:
        return graph

    for row in edge_df.itertuples(index=False):
        graph.add_edge(str(row.source_author), str(row.target_author), weight=int(row.weight))
    return graph


def _node_metrics(graph: nx.DiGraph) -> pd.DataFrame:
    if graph.number_of_nodes() == 0:
        return pd.DataFrame(
            columns=[
                "author_id",
                "in_degree",
                "out_degree",
                "weighted_in_degree",
                "weighted_out_degree",
                "undirected_clustering",
            ]
        )

    undirected = graph.to_undirected()
    clustering = nx.clustering(undirected)

    rows = []
    for node in graph.nodes():
        rows.append(
            {
                "author_id": node,
                "in_degree": int(graph.in_degree(node)),
                "out_degree": int(graph.out_degree(node)),
                "weighted_in_degree": float(graph.in_degree(node, weight="weight")),
                "weighted_out_degree": float(graph.out_degree(node, weight="weight")),
                "undirected_clustering": float(clustering.get(node, 0.0)),
            }
        )

    node_df = pd.DataFrame(rows)
    node_df = node_df.sort_values(
        by=["weighted_in_degree", "weighted_out_degree", "in_degree", "out_degree"],
        ascending=False,
    ).reset_index(drop=True)
    return node_df


def _thread_metrics(df: pd.DataFrame, thread_edge_df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "thread_id",
                "comment_count",
                "unique_authors",
                "max_level",
                "reply_edges",
                "reciprocal_author_pairs",
            ]
        )

    work = df.copy()
    work["thread_id"] = work["thread_id"].fillna("").astype(str)
    work["author_id"] = work["author_id"].fillna("").astype(str)
    work["level_num"] = pd.to_numeric(work.get("level", 0), errors="coerce").fillna(0)

    grouped = work.groupby("thread_id", dropna=False)

    base = grouped.agg(
        comment_count=("comment_id", "count"),
        unique_authors=("author_id", pd.Series.nunique),
        max_level=("level_num", "max"),
    ).reset_index()

    if thread_edge_df.empty:
        base["reply_edges"] = 0
        base["reciprocal_author_pairs"] = 0
        return base.sort_values("comment_count", ascending=False).reset_index(drop=True)

    edge_group = (
        thread_edge_df.groupby("thread_id", dropna=False)["weight"].sum().rename("reply_edges").reset_index()
    )

    reciprocal_rows: List[Dict[str, Any]] = []
    for thread_id, sub in thread_edge_df.groupby("thread_id", dropna=False):
        edge_set = {(str(r.source_author), str(r.target_author)) for r in sub.itertuples(index=False)}
        reciprocal_pair_count = 0
        for src, dst in edge_set:
            if src < dst and (dst, src) in edge_set:
                reciprocal_pair_count += 1
        reciprocal_rows.append(
            {
                "thread_id": thread_id,
                "reciprocal_author_pairs": int(reciprocal_pair_count),
            }
        )

    reciprocal_df = pd.DataFrame(reciprocal_rows)

    merged = base.merge(edge_group, on="thread_id", how="left").merge(
        reciprocal_df, on="thread_id", how="left"
    )
    merged["reply_edges"] = merged["reply_edges"].fillna(0).astype(int)
    merged["reciprocal_author_pairs"] = merged["reciprocal_author_pairs"].fillna(0).astype(int)

    return merged.sort_values("comment_count", ascending=False).reset_index(drop=True)


def _summarize_graph(
    graph: nx.DiGraph,
    resolution_counts: Dict[str, int],
    node_df: pd.DataFrame,
    selection_diagnostics: Dict[str, Any],
) -> Dict[str, Any]:
    node_count = int(graph.number_of_nodes())
    edge_count = int(graph.number_of_edges())

    weighted_interactions = 0
    if edge_count > 0:
        weighted_interactions = int(
            sum(int(data.get("weight", 1)) for _, _, data in graph.edges(data=True))
        )

    if node_count >= 2:
        density = float(nx.density(graph))
    else:
        density = 0.0

    if edge_count > 0:
        reciprocity_val = nx.reciprocity(graph)
        reciprocity = float(reciprocity_val) if reciprocity_val is not None else 0.0
    else:
        reciprocity = 0.0

    if node_count >= 2:
        avg_clustering = float(nx.average_clustering(graph.to_undirected()))
    else:
        avg_clustering = 0.0

    top_in = []
    top_out = []
    if not node_df.empty:
        top_in = (
            node_df.sort_values("weighted_in_degree", ascending=False)
            .head(10)[["author_id", "weighted_in_degree", "in_degree"]]
            .to_dict(orient="records")
        )
        top_out = (
            node_df.sort_values("weighted_out_degree", ascending=False)
            .head(10)[["author_id", "weighted_out_degree", "out_degree"]]
            .to_dict(orient="records")
        )

    resolved = int(resolution_counts.get("rows_parent_resolved_to_comment", 0))
    with_parent = int(resolution_counts.get("rows_with_parent_id", 0))
    coverage = float(resolved / with_parent) if with_parent > 0 else 0.0

    summary = {
        "graph_type": "directed_author_reply_graph",
        "edge_construction_mode": str(selection_diagnostics.get("selected_mode", "unknown")),
        "fallback_triggered": bool(selection_diagnostics.get("fallback_triggered", False)),
        "node_count": node_count,
        "edge_count": edge_count,
        "weighted_interactions": weighted_interactions,
        "density": round(density, 6),
        "reciprocity": round(reciprocity, 6),
        "average_undirected_clustering": round(avg_clustering, 6),
        "reply_resolution": {
            **resolution_counts,
            "reply_edge_coverage": round(coverage, 6),
        },
        "edge_selection_diagnostics": selection_diagnostics,
        "top_in_degree_authors": top_in,
        "top_out_degree_authors": top_out,
    }
    return summary


def _safe_label(text: str, max_len: int = 18) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _plot_top_network_snapshot(graph: nx.DiGraph, out_path: Path, top_n: int = 25) -> None:
    plt.figure(figsize=(12, 9))

    if graph.number_of_nodes() == 0:
        plt.text(0.5, 0.5, "No interaction edges to visualize", ha="center", va="center", fontsize=13)
        plt.axis("off")
        plt.title("Author Interaction Network (Top Nodes)")
        plt.tight_layout()
        plt.savefig(out_path, dpi=180)
        plt.close()
        return

    node_strength = {
        node: float(graph.in_degree(node, weight="weight") + graph.out_degree(node, weight="weight"))
        for node in graph.nodes()
    }
    top_nodes = [n for n, _ in sorted(node_strength.items(), key=lambda x: x[1], reverse=True)[:top_n]]
    sub = graph.subgraph(top_nodes).copy()

    if sub.number_of_edges() == 0:
        plt.text(0.5, 0.5, "Top nodes have no edges after filtering", ha="center", va="center", fontsize=12)
        plt.axis("off")
        plt.title("Author Interaction Network (Top Nodes)")
        plt.tight_layout()
        plt.savefig(out_path, dpi=180)
        plt.close()
        return

    pos = nx.spring_layout(sub, seed=42, k=0.65)

    strengths = [node_strength.get(n, 0.0) for n in sub.nodes()]
    max_strength = max(strengths) if strengths else 1.0
    node_sizes = [320 + 1700 * (s / max_strength) for s in strengths]

    edge_weights = [float(data.get("weight", 1.0)) for _, _, data in sub.edges(data=True)]
    max_w = max(edge_weights) if edge_weights else 1.0
    edge_widths = [0.7 + 4.3 * (w / max_w) for w in edge_weights]

    nx.draw_networkx_nodes(
        sub,
        pos,
        node_size=node_sizes,
        node_color="#88bde6",
        edgecolors="#1d4e89",
        linewidths=0.8,
        alpha=0.95,
    )
    nx.draw_networkx_edges(
        sub,
        pos,
        width=edge_widths,
        edge_color="#4f81bd",
        alpha=0.55,
        arrows=True,
        arrowsize=11,
        arrowstyle="-|>",
        min_source_margin=8,
        min_target_margin=8,
        connectionstyle="arc3,rad=0.12",
    )

    labels = {n: _safe_label(str(n), 20) for n in sub.nodes()}
    nx.draw_networkx_labels(sub, pos, labels=labels, font_size=8)

    plt.title("Author Interaction Network (Top Nodes by Weighted Degree)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_metric_distributions(node_df: pd.DataFrame, thread_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    if node_df.empty:
        axes[0].text(0.5, 0.5, "No node data", ha="center", va="center")
        axes[1].text(0.5, 0.5, "No node data", ha="center", va="center")
    else:
        axes[0].hist(node_df["in_degree"].astype(float), bins=20, color="#7cb5ec", edgecolor="white")
        axes[0].set_title("In-Degree Distribution")
        axes[0].set_xlabel("In-Degree")
        axes[0].set_ylabel("Authors")

        axes[1].hist(node_df["out_degree"].astype(float), bins=20, color="#90ed7d", edgecolor="white")
        axes[1].set_title("Out-Degree Distribution")
        axes[1].set_xlabel("Out-Degree")
        axes[1].set_ylabel("Authors")

    if thread_df.empty:
        axes[2].text(0.5, 0.5, "No thread data", ha="center", va="center")
    else:
        vals = thread_df["reply_edges"].astype(float)
        axes[2].hist(vals, bins=20, color="#f7a35c", edgecolor="white")
        axes[2].set_title("Reply Edges per Thread")
        axes[2].set_xlabel("Reply Edges")
        axes[2].set_ylabel("Threads")

    plt.suptitle("Interaction Network Metric Distributions")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _write_manifest(
    run_id: str,
    input_file: Path,
    output_files: List[Path],
    edge_mode: str,
    edge_mode_reason: str,
    seed: int,
) -> Path:
    manifest_dir = REPO_ROOT / "data" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "experiment": "rq1_interaction_network",
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "input_file": input_file.as_posix(),
        "output_files": [p.as_posix() for p in output_files],
        "parameters": {
            "seed": int(seed),
            "edge_mode": edge_mode,
        },
        "edge_mode_reason": edge_mode_reason,
    }

    out_path = manifest_dir / f"interaction_network_manifest_{run_id}.json"
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build directed author reply network and export RQ1 interaction metrics."
    )
    parser.add_argument(
        "--input",
        default="data/staged/moltbook_comments_all.jsonl",
        help="Input staged comments JSONL file.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/eda",
        help="Directory for network outputs.",
    )
    parser.add_argument(
        "--edge-mode",
        default="auto",
        choices=["auto", "direct", "sequential"],
        help=(
            "Edge construction strategy: auto (prefer direct parent replies, fallback to sequential), "
            "direct (parent->child comment links only), or sequential (adjacent comments within thread)."
        ),
    )
    parser.add_argument(
        "--edge-mode-justification",
        action="store_true",
        help="Print a human-readable explanation of direct/sequential edge logic and exit.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Optional run ID (UTC format YYYYMMDDTHHMMSSZ). Defaults to current UTC time.",
    )
    args = parser.parse_args()

    if args.edge_mode_justification:
        print(EDGE_MODE_JUSTIFICATION.strip())
        return

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = args.run_id.strip() or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    rows = _read_jsonl(input_path)
    df = pd.DataFrame(rows)

    required_cols = [
        "comment_id",
        "post_id",
        "thread_id",
        "parent_id",
        "author_id",
        "level",
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = "" if col != "level" else 0

    df["comment_id"] = df["comment_id"].fillna("").astype(str)
    df["thread_id"] = df["thread_id"].fillna("").astype(str)
    df["parent_id"] = df["parent_id"].fillna("").astype(str)
    df["author_id"] = df["author_id"].fillna("").astype(str)

    edge_df, thread_edge_df, resolution_counts, selection_diagnostics = _select_edge_tables(
        df, args.edge_mode
    )
    graph = _build_graph(edge_df)
    node_df = _node_metrics(graph)
    thread_df = _thread_metrics(df, thread_edge_df)
    summary = _summarize_graph(graph, resolution_counts, node_df, selection_diagnostics)

    summary.update(
        {
            "run_id": run_id,
            "input_file": str(input_path).replace("\\", "/"),
            "rows_input": int(len(df)),
            "edge_mode_justification": EDGE_MODE_JUSTIFICATION.strip(),
        }
    )

    summary_path = output_dir / f"moltbook_interaction_network_summary_{run_id}.json"
    nodes_path = output_dir / f"moltbook_interaction_network_nodes_{run_id}.csv"
    edges_path = output_dir / f"moltbook_interaction_network_edges_{run_id}.csv"
    thread_path = output_dir / f"moltbook_interaction_network_thread_stats_{run_id}.csv"
    network_plot_path = output_dir / f"moltbook_interaction_network_topology_{run_id}.png"
    distribution_plot_path = output_dir / f"moltbook_interaction_network_distributions_{run_id}.png"
    latest_network_plot_path = output_dir / "moltbook_interaction_network_topology_latest.png"
    latest_distribution_plot_path = output_dir / "moltbook_interaction_network_distributions_latest.png"

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    node_df.to_csv(nodes_path, index=False, encoding="utf-8")
    edge_df.to_csv(edges_path, index=False, encoding="utf-8")
    thread_df.to_csv(thread_path, index=False, encoding="utf-8")
    _plot_top_network_snapshot(graph, network_plot_path, top_n=25)
    _plot_metric_distributions(node_df, thread_df, distribution_plot_path)
    shutil.copyfile(network_plot_path, latest_network_plot_path)
    shutil.copyfile(distribution_plot_path, latest_distribution_plot_path)

    selected = str(selection_diagnostics.get("selected_mode", "unknown"))
    reason = (
        "auto mode selected direct edges because parent-child links were resolvable"
        if selected == "direct"
        else "sequential fallback selected because direct parent-child linkage did not resolve to in-corpus comment edges"
    )
    manifest_path = _write_manifest(
        run_id=run_id,
        input_file=input_path,
        output_files=[
            summary_path,
            nodes_path,
            edges_path,
            thread_path,
            network_plot_path,
            distribution_plot_path,
            latest_network_plot_path,
            latest_distribution_plot_path,
        ],
        edge_mode=selected,
        edge_mode_reason=reason,
        seed=args.seed,
    )

    cleanup_old_files(output_dir, "moltbook_interaction_network_summary_*.json", keep_latest=1)
    cleanup_old_files(output_dir, "moltbook_interaction_network_nodes_*.csv", keep_latest=1)
    cleanup_old_files(output_dir, "moltbook_interaction_network_edges_*.csv", keep_latest=1)
    cleanup_old_files(output_dir, "moltbook_interaction_network_thread_stats_*.csv", keep_latest=1)
    cleanup_old_files(output_dir, "moltbook_interaction_network_topology_20*.png", keep_latest=1)
    cleanup_old_files(output_dir, "moltbook_interaction_network_distributions_20*.png", keep_latest=1)

    print("Interaction network analysis complete")
    print(f"input_file: {input_path}")
    print(f"rows_input: {len(df)}")
    print(f"nodes: {summary['node_count']}")
    print(f"edges: {summary['edge_count']}")
    print(f"weighted_interactions: {summary['weighted_interactions']}")
    print(f"edge_construction_mode: {summary['edge_construction_mode']}")
    print(f"fallback_triggered: {summary['fallback_triggered']}")
    print(f"reciprocity: {summary['reciprocity']:.6f}")
    print(f"average_undirected_clustering: {summary['average_undirected_clustering']:.6f}")
    print(f"reply_edge_coverage: {summary['reply_resolution']['reply_edge_coverage']:.6f}")
    print(f"summary_path: {summary_path}")
    print(f"nodes_path: {nodes_path}")
    print(f"edges_path: {edges_path}")
    print(f"thread_stats_path: {thread_path}")
    print(f"network_plot_path: {network_plot_path}")
    print(f"distribution_plot_path: {distribution_plot_path}")
    print(f"network_plot_latest_path: {latest_network_plot_path}")
    print(f"distribution_plot_latest_path: {latest_distribution_plot_path}")
    print(f"manifest_path: {manifest_path}")


if __name__ == "__main__":
    main()
