"""TTC structure-efficiency experiment for section 8.4."""

from __future__ import annotations

import os
import random
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
TYPE_I_DIR = PROJECT_ROOT / "evaluation" / "type_I"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "8.4"

if str(TYPE_I_DIR) not in sys.path:
    sys.path.insert(0, str(TYPE_I_DIR))

from graphBuilder import build_graph


warnings.filterwarnings("ignore")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_SAMPLE_ROWS = 500_000
NUM_TRIALS = 8
RANDOM_SEED = 42
TOP_HUB_N = 120

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def make_weight_fn():
    """Count edge inspections during Dijkstra."""

    counter = {"edges_checked": 0}

    def weight_fn(u, v, edge_data):
        counter["edges_checked"] += 1
        return edge_data.get("weight", 1)

    return weight_fn, counter


def path_cost(graph, path):
    """Sum the edge weights along a returned path."""
    return sum(graph[path[i]][path[i + 1]].get("weight", 1) for i in range(len(path) - 1))


def run_dijkstra(graph, source, target):
    """Run Dijkstra and return path metrics."""
    weight_fn, counter = make_weight_fn()
    start = time.perf_counter()
    path = nx.dijkstra_path(graph, source, target, weight=weight_fn)
    elapsed = time.perf_counter() - start
    return {
        "path": path,
        "path_cost": path_cost(graph, path),
        "hops": len(path) - 1,
        "edges_evaluated": counter["edges_checked"],
        "runtime_s": elapsed,
    }


def sample_reachable_pairs(graph, num_trials, seed):
    """Sample reachable source-target pairs from a graph."""
    rng = random.Random(seed)
    nodes = list(graph.nodes())
    pairs = []
    attempts = 0
    max_attempts = num_trials * 300

    while len(pairs) < num_trials and attempts < max_attempts:
        attempts += 1
        source, target = rng.sample(nodes, 2)
        if nx.has_path(graph, source, target):
            pairs.append((source, target))

    return pairs


def graph_overview(graph):
    """Return structural summary rows for a graph."""
    if graph.is_directed():
        components = nx.number_weakly_connected_components(graph)
    else:
        components = nx.number_connected_components(graph)

    avg_degree = (
        sum(deg for _, deg in graph.degree()) / graph.number_of_nodes()
        if graph.number_of_nodes()
        else 0
    )

    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "density": nx.density(graph),
        "components": components,
        "avg_degree": avg_degree,
    }


def build_hub_component(graph, top_n=TOP_HUB_N):
    """Build a hub-focused subgraph from the highest-degree stops and their neighbors."""
    top_nodes = [
        node
        for node, _ in sorted(graph.degree(), key=lambda item: item[1], reverse=True)[:top_n]
    ]
    hub_nodes = set(top_nodes)

    for node in top_nodes:
        if graph.is_directed():
            hub_nodes.update(graph.predecessors(node))
            hub_nodes.update(graph.successors(node))
        else:
            hub_nodes.update(graph.neighbors(node))

    hub_subgraph = graph.subgraph(hub_nodes).copy()

    if hub_subgraph.number_of_nodes() == 0:
        return hub_subgraph

    if hub_subgraph.is_directed():
        components = list(nx.weakly_connected_components(hub_subgraph))
    else:
        components = list(nx.connected_components(hub_subgraph))

    if not components:
        return hub_subgraph

    largest = max(components, key=len)
    return hub_subgraph.subgraph(largest).copy()


def benchmark_structure(name, graph, seed_offset):
    """Run the routing benchmark for one graph structure."""
    print(f"\nBenchmarking {name} ...")
    overview = graph_overview(graph)
    pairs = sample_reachable_pairs(graph, NUM_TRIALS, RANDOM_SEED + seed_offset)

    if len(pairs) < NUM_TRIALS:
        print(f"  warning: only found {len(pairs)} reachable pairs for {name}")

    rows = []
    for trial_num, (source, target) in enumerate(pairs, start=1):
        result = run_dijkstra(graph, source, target)
        rows.append(
            {
                "structure": name,
                "trial": trial_num,
                "source_id": source,
                "source_name": graph.nodes[source].get("name", str(source)),
                "target_id": target,
                "target_name": graph.nodes[target].get("name", str(target)),
                "runtime_s": result["runtime_s"],
                "edges_evaluated": result["edges_evaluated"],
                "path_cost": result["path_cost"],
                "hops": result["hops"],
            }
        )

    return overview, pd.DataFrame(rows)


def plot_summary(summary_df):
    """Create a two-panel chart for structure comparison."""
    structure_order = list(summary_df["structure"].unique())
    x = np.arange(len(structure_order))

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle("TTC Graph Structures: Computational Efficiency Comparison", fontsize=14, fontweight="bold")

    runtime_ms = (
        summary_df.set_index("structure")
        .reindex(structure_order)["avg_runtime_s"]
        .values
        * 1000
    )
    search_effort = summary_df.set_index("structure").reindex(structure_order)["avg_edges_evaluated"].values

    panels = [
        (axes[0], runtime_ms, "Runtime (ms)", "Average runtime"),
        (axes[1], search_effort, "Edges evaluated", "Average search effort"),
    ]

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    for ax, values, ylabel, title in panels:
        bars = ax.bar(x, values, color=colors[: len(structure_order)], alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(structure_order, rotation=18, ha="right")
        ax.set_xlabel("Graph structure")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.2f}" if ylabel == "Runtime (ms)" else f"{height:.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    out_path = OUTPUT_DIR / "structure_efficiency_chart.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved {out_path}")


def main():
    print("=" * 70)
    print("  TTC Structure Efficiency Study (Section 8.4)")
    print("=" * 70)

    print("\nLoading baseline TTC graph ...")
    full_graph, giant_graph = build_graph(sample_rows=BASE_SAMPLE_ROWS)
    undirected_giant = giant_graph.to_undirected()
    hub_component = build_hub_component(giant_graph, top_n=TOP_HUB_N)

    structures = [
        ("Directed full graph", full_graph),
        ("Largest weakly connected component", giant_graph),
        ("Undirected giant component", undirected_giant),
        ("Hub neighborhood component", hub_component),
    ]

    graph_rows = []
    trial_frames = []

    for idx, (name, graph) in enumerate(structures, start=1):
        overview, trials = benchmark_structure(name, graph, seed_offset=idx * 1000)
        overview["structure"] = name
        overview["trials_found"] = len(trials)
        graph_rows.append(overview)
        trial_frames.append(trials)

    graphs_df = pd.DataFrame(graph_rows)
    results_df = pd.concat(trial_frames, ignore_index=True)

    summary_df = (
        results_df.groupby("structure", as_index=False)
        .agg(
            avg_runtime_s=("runtime_s", "mean"),
            std_runtime_s=("runtime_s", "std"),
            avg_edges_evaluated=("edges_evaluated", "mean"),
            std_edges_evaluated=("edges_evaluated", "std"),
            avg_path_cost=("path_cost", "mean"),
            avg_hops=("hops", "mean"),
        )
        .merge(graphs_df, on="structure", how="left")
    )

    structure_order = [name for name, _ in structures]
    summary_df["structure"] = pd.Categorical(summary_df["structure"], categories=structure_order, ordered=True)
    summary_df = summary_df.sort_values("structure").reset_index(drop=True)

    graphs_csv = OUTPUT_DIR / "structure_sizes.csv"
    trials_csv = OUTPUT_DIR / "structure_trials.csv"
    summary_csv = OUTPUT_DIR / "structure_summary.csv"

    graphs_df.to_csv(graphs_csv, index=False)
    results_df.to_csv(trials_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    print("\nStructure summary")
    print(
        graphs_df[
            ["structure", "nodes", "edges", "components", "density", "avg_degree", "trials_found"]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    print("\nEfficiency summary")
    print(
        summary_df[
            ["structure", "avg_runtime_s", "avg_edges_evaluated", "avg_path_cost", "avg_hops"]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    plot_summary(summary_df)

    print("\nSaved tables")
    print(f"  {graphs_csv}")
    print(f"  {trials_csv}")
    print(f"  {summary_csv}")
    print("\nDone.")


if __name__ == "__main__":
    main()
