"""TTC scalability experiment for section 8.3."""

from __future__ import annotations

import math
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
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "8.3"

if str(TYPE_I_DIR) not in sys.path:
    sys.path.insert(0, str(TYPE_I_DIR))

from graphBuilder import build_graph


warnings.filterwarnings("ignore")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GRAPH_SIZES = [
    (100_000, "100k rows"),
    (250_000, "250k rows"),
    (500_000, "500k rows"),
]
NUM_TRIALS = 8
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in kilometres."""
    radius_km = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    )
    return radius_km * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def make_astar_heuristic(graph, target):
    """Heuristic used by A*; scaled so it stays admissible for trip-frequency weights."""

    def heuristic(node, goal):
        try:
            node_data = graph.nodes[node]
            target_data = graph.nodes[target]
            return haversine_km(
                node_data["lat"],
                node_data["lon"],
                target_data["lat"],
                target_data["lon"],
            ) / 10000
        except (KeyError, TypeError):
            return 0.0

    return heuristic


def make_weight_fn():
    """Count how many edges the algorithm inspects."""

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
        "algorithm": "Dijkstra",
        "path": path,
        "path_cost": path_cost(graph, path),
        "hops": len(path) - 1,
        "edges_evaluated": counter["edges_checked"],
        "runtime_s": elapsed,
    }


def run_astar(graph, source, target):
    """Run A* and return path metrics."""
    weight_fn, counter = make_weight_fn()
    start = time.perf_counter()
    path = nx.astar_path(
        graph,
        source,
        target,
        heuristic=make_astar_heuristic(graph, target),
        weight=weight_fn,
    )
    elapsed = time.perf_counter() - start
    return {
        "algorithm": "A*",
        "path": path,
        "path_cost": path_cost(graph, path),
        "hops": len(path) - 1,
        "edges_evaluated": counter["edges_checked"],
        "runtime_s": elapsed,
    }


def sample_reachable_pairs(graph, num_trials, seed):
    """Sample reachable source-target pairs from the graph."""
    rng = random.Random(seed)
    nodes = list(graph.nodes())
    pairs = []
    attempts = 0
    max_attempts = num_trials * 250

    while len(pairs) < num_trials and attempts < max_attempts:
        attempts += 1
        source, target = rng.sample(nodes, 2)
        if nx.has_path(graph, source, target):
            pairs.append((source, target))

    return pairs


def graph_overview(full_graph, giant_graph, sample_rows, size_label):
    """Return structural summary rows for the graph sample."""
    if full_graph.is_directed():
        components = nx.number_weakly_connected_components(full_graph)
    else:
        components = nx.number_connected_components(full_graph)

    avg_degree = (
        sum(deg for _, deg in giant_graph.degree()) / giant_graph.number_of_nodes()
        if giant_graph.number_of_nodes()
        else 0
    )

    return {
        "sample_rows": sample_rows,
        "size_label": size_label,
        "full_nodes": full_graph.number_of_nodes(),
        "full_edges": full_graph.number_of_edges(),
        "giant_nodes": giant_graph.number_of_nodes(),
        "giant_edges": giant_graph.number_of_edges(),
        "components": components,
        "giant_fraction_pct": 100 * giant_graph.number_of_nodes() / full_graph.number_of_nodes(),
        "giant_density": nx.density(giant_graph),
        "giant_avg_degree": avg_degree,
    }


def benchmark_sample(sample_rows, size_label, seed_offset):
    """Build one graph size and benchmark both shortest-path algorithms."""
    print(f"\nBuilding {size_label} graph ...")
    full_graph, giant_graph = build_graph(sample_rows=sample_rows)
    print(
        f"  graph stats: {full_graph.number_of_nodes():,} nodes, "
        f"{full_graph.number_of_edges():,} edges"
    )
    print(
        f"  giant comp. : {giant_graph.number_of_nodes():,} nodes, "
        f"{giant_graph.number_of_edges():,} edges"
    )

    overview = graph_overview(full_graph, giant_graph, sample_rows, size_label)
    pairs = sample_reachable_pairs(giant_graph, NUM_TRIALS, RANDOM_SEED + seed_offset)

    if len(pairs) < NUM_TRIALS:
        print(f"  warning: only found {len(pairs)} reachable pairs for {size_label}")

    trial_rows = []
    for trial_num, (source, target) in enumerate(pairs, start=1):
        for runner in (run_dijkstra, run_astar):
            result = runner(giant_graph, source, target)
            trial_rows.append(
                {
                    "sample_rows": sample_rows,
                    "size_label": size_label,
                    "trial": trial_num,
                    "source_id": source,
                    "source_name": giant_graph.nodes[source].get("name", str(source)),
                    "target_id": target,
                    "target_name": giant_graph.nodes[target].get("name", str(target)),
                    "algorithm": result["algorithm"],
                    "runtime_s": result["runtime_s"],
                    "edges_evaluated": result["edges_evaluated"],
                    "path_cost": result["path_cost"],
                    "hops": result["hops"],
                }
            )

    return overview, pd.DataFrame(trial_rows)


def print_table(title, df, columns):
    """Print a compact table to the terminal."""
    print("\n" + title)
    print(df[columns].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def plot_summary(summary_df):
    """Create a two-panel chart for runtime and search effort."""
    size_order = [label for _, label in GRAPH_SIZES]
    alg_order = ["Dijkstra", "A*"]
    colors = {"Dijkstra": "#4C72B0", "A*": "#DD8452"}
    x = np.arange(len(size_order))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("TTC Scalability Study: Graph Size vs Search Cost", fontsize=14, fontweight="bold")

    for ax, metric, ylabel, title in [
        (axes[0], "avg_runtime_s", "Runtime (seconds)", "Average runtime by graph size"),
        (axes[1], "avg_edges_evaluated", "Edges evaluated", "Average search effort by graph size"),
    ]:
        for offset, algo in zip([-width / 2, width / 2], alg_order):
            subset = (
                summary_df[summary_df["algorithm"] == algo]
                .set_index("size_label")
                .reindex(size_order)
            )
            ax.bar(
                x + offset,
                subset[metric].values,
                width,
                label=algo,
                color=colors[algo],
                alpha=0.9,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(size_order)
        ax.set_xlabel("GTFS sample size")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.legend()

    plt.tight_layout()
    out_path = OUTPUT_DIR / "scalability_chart.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved {out_path}")


def main():
    print("=" * 70)
    print("  TTC Scalability Study (Section 8.3)")
    print("=" * 70)

    graph_rows = []
    trial_frames = []

    for idx, (sample_rows, size_label) in enumerate(GRAPH_SIZES, start=1):
        overview, trials = benchmark_sample(sample_rows, size_label, seed_offset=idx * 1000)
        graph_rows.append(overview)
        trial_frames.append(trials)

    graphs_df = pd.DataFrame(graph_rows)
    results_df = pd.concat(trial_frames, ignore_index=True)

    summary_df = (
        results_df.groupby(["sample_rows", "size_label", "algorithm"], as_index=False)
        .agg(
            avg_runtime_s=("runtime_s", "mean"),
            std_runtime_s=("runtime_s", "std"),
            avg_edges_evaluated=("edges_evaluated", "mean"),
            std_edges_evaluated=("edges_evaluated", "std"),
            avg_path_cost=("path_cost", "mean"),
            avg_hops=("hops", "mean"),
        )
        .merge(graphs_df, on=["sample_rows", "size_label"], how="left")
    )

    size_order = [label for _, label in GRAPH_SIZES]
    summary_df["size_label"] = pd.Categorical(summary_df["size_label"], categories=size_order, ordered=True)
    summary_df["algorithm"] = pd.Categorical(summary_df["algorithm"], categories=["Dijkstra", "A*"], ordered=True)
    summary_df = summary_df.sort_values(["size_label", "algorithm"]).reset_index(drop=True)

    graphs_csv = OUTPUT_DIR / "graph_sizes.csv"
    trials_csv = OUTPUT_DIR / "scalability_trials.csv"
    summary_csv = OUTPUT_DIR / "scalability_summary.csv"

    graphs_df.to_csv(graphs_csv, index=False)
    results_df.to_csv(trials_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    print("\nGraph size summary")
    print(
        graphs_df[
            [
                "size_label",
                "full_nodes",
                "full_edges",
                "giant_nodes",
                "giant_edges",
                "components",
                "giant_fraction_pct",
                "giant_density",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    print("\nEfficiency summary")
    print(
        summary_df[
            [
                "size_label",
                "algorithm",
                "avg_runtime_s",
                "avg_edges_evaluated",
                "avg_path_cost",
                "avg_hops",
            ]
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
