"""
TTC Shortest-Path Routing: Dijkstra vs A*
======================================================
Compares Dijkstra's algorithm and A* search on the TTC stop network.
Graph construction is delegated to graph_builder.py so the same graph
object is shared with type1_abrar.py.

Both algorithms are applied to several randomly-selected (source, destination)
pairs drawn from the giant connected component of the graph.  For each trial
the script records:
  - runtime (seconds)
  - edges evaluated during the search
  - total path cost (sum of edge weights along the returned path)
  - path length (number of hops)

A* uses the Haversine geographic distance between stops as its admissible
heuristic — this never overestimates the true cost, so A* is guaranteed to
return the same optimal path as Dijkstra.

Run from the project root (ttc-network-analysis/):
  python evaluation/type_I/type1_routing.py

Outputs
-------
  outputs/routing_comparison.csv   — per-trial results table
  outputs/routing_summary.png      — bar-chart summary (runtime & nodes)
  outputs/routing_paths.png        — sample path visualised on the map
"""

import math
import os
import random
import time
import warnings

import matplotlib
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

# ── Shared graph ──────────────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from graphBuilder import build_graph

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

NUM_TRIALS  = 10    # number of random source→destination pairs to test
RANDOM_SEED = 42    # reproducibility
OUTPUT_DIR  = "outputs"

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("  EECS4414 — TTC Routing: Dijkstra vs A*")
print("=" * 60)
print("\nLoading graph …")
G, giant = build_graph()
giant_nodes = list(giant.nodes())
print()


# ══════════════════════════════════════════════════════════════════════════════
# HEURISTIC FOR A*
# ══════════════════════════════════════════════════════════════════════════════
# Haversine distance (km) between two stops.
# Edge weights are trip-frequency counts, not kilometres, so the heuristic
# underestimates real cost — making it admissible and guaranteeing optimality.
# ──────────────────────────────────────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in kilometres."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def astar_heuristic(u, v):
    """
    Admissible heuristic: straight-line geographic distance.
    Returns 0 if coordinate data is unavailable (safe fallback).
    """
    try:
        ud, vd = giant.nodes[u], giant.nodes[v]
        return haversine_km(ud["lat"], ud["lon"], vd["lat"], vd["lon"])
    except (KeyError, TypeError):
        return 0.0



# ══════════════════════════════════════════════════════════════════════════════
# EDGE-VISIT COUNTER
# ══════════════════════════════════════════════════════════════════════════════
# NetworkX's Dijkstra/A* internals access the graph at the C level, so a
# Python proxy around the graph object is never actually called.
#
# Instead we pass a *weight function* — both nx.dijkstra_path and nx.astar_path
# accept  weight=callable(u, v, edge_data) -> float  in place of a string key.
# Every time the algorithm evaluates an edge it calls our function, so counting
# those calls gives a reliable measure of edges (and therefore nodes) examined.
# ──────────────────────────────────────────────────────────────────────────────

def make_weight_fn():
    """
    Returns (weight_fn, counter).
    weight_fn(u, v, data) returns the edge weight and increments counter['n'].
    """
    counter = {"n": 0}

    def weight_fn(u, v, data):
        counter["n"] += 1
        return data.get("weight", 1)

    return weight_fn, counter


# ──────────────────────────────────────────────────────────────────────────────
# Algorithm runners
# ──────────────────────────────────────────────────────────────────────────────

def run_dijkstra(graph, source, target):
    """
    Run nx.dijkstra_path with an instrumented weight function.
    Returns (path, path_cost, edges_evaluated, elapsed_seconds).
    """
    wfn, ctr = make_weight_fn()
    t0       = time.perf_counter()
    path     = nx.dijkstra_path(graph, source, target, weight=wfn)
    elapsed  = time.perf_counter() - t0
    cost     = nx.dijkstra_path_length(graph, source, target, weight="weight")
    return path, cost, ctr["n"], elapsed


def run_astar(graph, source, target):
    """
    Run nx.astar_path with an instrumented weight function.
    Returns (path, path_cost, edges_evaluated, elapsed_seconds).
    """
    wfn, ctr = make_weight_fn()
    t0       = time.perf_counter()
    path     = nx.astar_path(graph, source, target,
                             heuristic=astar_heuristic, weight=wfn)
    elapsed  = time.perf_counter() - t0
    cost     = sum(
        graph[path[i]][path[i + 1]]["weight"]
        for i in range(len(path) - 1)
    )
    return path, cost, ctr["n"], elapsed


# ══════════════════════════════════════════════════════════════════════════════
# RUN TRIALS
# ══════════════════════════════════════════════════════════════════════════════

print(f"Step 4 — Running {NUM_TRIALS} routing trials …\n")

# Sample reachable pairs
trial_pairs, attempts = [], 0
while len(trial_pairs) < NUM_TRIALS and attempts < NUM_TRIALS * 20:
    attempts += 1
    src, dst = random.sample(giant_nodes, 2)
    if nx.has_path(giant, src, dst):
        trial_pairs.append((src, dst))

if len(trial_pairs) < NUM_TRIALS:
    print(f"  Warning: only found {len(trial_pairs)} reachable pairs.\n")

# Print header
header = (
    f"{'#':<4} {'Algorithm':<10} {'Runtime (s)':<13} "
    f"{'Edges Eval.':<12} {'Path Cost':<12} {'Hops':<6} "
    f"{'Source':<30} {'→  Destination'}"
)
print(header)
print("─" * len(header))

results = []

for i, (src, dst) in enumerate(trial_pairs, 1):
    src_name = giant.nodes[src].get("name", str(src))[:28]
    dst_name = giant.nodes[dst].get("name", str(dst))[:28]

    try:
        d_path, d_cost, d_nodes, d_time = run_dijkstra(giant, src, dst)
        a_path, a_cost, a_nodes, a_time = run_astar(giant, src, dst)

        same_cost = abs(d_cost - a_cost) < 1e-6

        for algo, path, cost, nodes, rt in [
            ("Dijkstra", d_path, d_cost, d_nodes, d_time),
            ("A*",       a_path, a_cost, a_nodes, a_time),
        ]:
            print(
                f"{i:<4} {algo:<10} {rt:<13.6f} {nodes:<12} "
                f"{cost:<12.1f} {len(path)-1:<6} {src_name:<30}    {dst_name}"
            )
            results.append({
                "trial":          i,
                "source_id":      src,
                "source_name":    src_name,
                "dest_id":        dst,
                "dest_name":      dst_name,
                "algorithm":      algo,
                "runtime_s":      round(rt, 6),
                "edges_evaluated": nodes,
                "path_cost":      round(cost, 2),
                "hops":           len(path) - 1,
                "same_cost":      same_cost,
            })

    except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
        print(f"  Trial {i} skipped: {e}")

print()


# ══════════════════════════════════════════════════════════════════════════════
# SAVE CSV
# ══════════════════════════════════════════════════════════════════════════════

results_df = pd.DataFrame(results)
csv_path   = f"{OUTPUT_DIR}/routing_comparison.csv"
results_df.to_csv(csv_path, index=False)
print(f"  → Results saved to {csv_path}")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE  (matches Table 3 format in the report)
# ══════════════════════════════════════════════════════════════════════════════

summary = (
    results_df
    .groupby("algorithm")
    .agg(
        avg_runtime_s      =("runtime_s",       "mean"),
        avg_edges_evaluated =("edges_evaluated",   "mean"),
        avg_path_cost      =("path_cost",        "mean"),
        avg_hops           =("hops",             "mean"),
    )
    .reset_index()
)

print("\n" + "═" * 66)
print("  SUMMARY TABLE — Dijkstra vs A* on TTC Network")
print("═" * 66)
print(f"\n  Trials : {NUM_TRIALS}    |    "
      f"Graph : {giant.number_of_nodes():,} nodes, {giant.number_of_edges():,} edges\n")

col_w   = [12, 16, 16, 14, 10]
headers = ["Algorithm", "Edge Weight", "Avg Runtime (s)", "Avg Path Cost", "Avg Hops"]
row_fmt = "  " + "  ".join(f"{{:<{w}}}" for w in col_w)

print(row_fmt.format(*headers))
print("  " + "─" * (sum(col_w) + 2 * len(col_w)))

for _, row in summary.iterrows():
    print(row_fmt.format(
        row["algorithm"],
        "trip frequency",
        f"{row['avg_runtime_s']:.6f}",
        f"{row['avg_path_cost']:.1f}",
        f"{row['avg_hops']:.1f}",
    ))

print()
same_count = results_df.groupby("trial")["path_cost"].nunique().eq(1).sum()
print(f"  Trials where both algorithms returned identical path cost: "
      f"{same_count}/{len(trial_pairs)}")


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════

print("\nStep 5 — Generating plots …")

dijk_df = results_df[results_df["algorithm"] == "Dijkstra"]
astr_df = results_df[results_df["algorithm"] == "A*"]
trials  = dijk_df["trial"].values
x       = np.arange(len(trials))
bar_w   = 0.35

# ── 5a. Runtime and nodes-explored bar charts ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Dijkstra vs A* — TTC Network Routing Comparison",
             fontsize=14, fontweight="bold")

ax = axes[0]
ax.bar(x - bar_w / 2, dijk_df["runtime_s"].values, bar_w, label="Dijkstra", color="#4C72B0")
ax.bar(x + bar_w / 2, astr_df["runtime_s"].values, bar_w, label="A*",       color="#DD8452")
ax.set_xlabel("Trial")
ax.set_ylabel("Runtime (seconds)")
ax.set_title("Runtime per Trial")
ax.set_xticks(x)
ax.set_xticklabels([f"T{t}" for t in trials])
ax.legend()
ax.grid(axis="y", alpha=0.3)

ax = axes[1]
ax.bar(x - bar_w / 2, dijk_df["edges_evaluated"].values, bar_w, label="Dijkstra", color="#4C72B0")
ax.bar(x + bar_w / 2, astr_df["edges_evaluated"].values, bar_w, label="A*",       color="#DD8452")
ax.set_xlabel("Trial")
ax.set_ylabel("Edges Evaluated (proxy for nodes explored)")
ax.set_title("Edges Evaluated per Trial")
ax.set_xticks(x)
ax.set_xticklabels([f"T{t}" for t in trials])
ax.legend()
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/routing_summary.png", dpi=150)
plt.close()
print(f"  → {OUTPUT_DIR}/routing_summary.png saved")


# ── 5b. One map per trial saved into outputs/routing_paths/ ──────────────────
PATHS_DIR = f"{OUTPUT_DIR}/routing_paths"
os.makedirs(PATHS_DIR, exist_ok=True)

# Pre-compute geographic positions once (reused for every trial)
pos_geo = {
    n: (giant.nodes[n]["lon"], giant.nodes[n]["lat"])
    for n in giant.nodes()
    if "lon" in giant.nodes[n] and "lat" in giant.nodes[n]
}

def draw_path_on_ax(ax, path, color, lw=2.5):
    path_edges = list(zip(path, path[1:]))
    path_pos   = {n: pos_geo[n] for n in path if n in pos_geo}
    nx.draw_networkx_nodes(
        giant, path_pos, nodelist=list(path_pos.keys()),
        node_size=30, node_color=color, alpha=0.9, ax=ax,
    )
    nx.draw_networkx_edges(
        giant, pos_geo, edgelist=path_edges,
        edge_color=color, width=lw, alpha=0.8,
        arrows=True, arrowsize=10, ax=ax,
    )

for trial_num, (src_id, dst_id) in enumerate(trial_pairs, 1):
    try:
        d_path, _, _, _ = run_dijkstra(giant, src_id, dst_id)
        a_path, _, _, _ = run_astar(giant,   src_id, dst_id)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        continue

    fig, ax = plt.subplots(figsize=(12, 10))

    # Background: all nodes faint
    nx.draw_networkx_nodes(
        giant, pos_geo,
        nodelist=list(pos_geo.keys()),
        node_size=2, node_color="#cccccc", alpha=0.4, ax=ax,
    )

    draw_path_on_ax(ax, d_path, "#4C72B0", lw=3)
    draw_path_on_ax(ax, a_path, "#DD8452", lw=1.5)

    for nid, label, color in [(src_id, "SOURCE", "green"), (dst_id, "DEST", "red")]:
        if nid in pos_geo:
            ax.annotate(
                f"{label}\n{giant.nodes[nid].get('name', nid)[:30]}",
                xy=pos_geo[nid], fontsize=7, color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
            )

    legend_patches = [
        mpatches.Patch(color="#4C72B0", label="Dijkstra path"),
        mpatches.Patch(color="#DD8452", label="A* path"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9)
    ax.set_title(
        f"Routing Trial {trial_num}\n"
        f"Source: {giant.nodes[src_id].get('name','?')[:40]}  \u2192  "
        f"Dest:   {giant.nodes[dst_id].get('name','?')[:40]}",
        fontsize=10,
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()

    out_path = f"{PATHS_DIR}/trial_{trial_num:02d}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  \u2192 {out_path} saved")

print(f"  All {len(trial_pairs)} trial maps saved to {PATHS_DIR}/")

# ══════════════════════════════════════════════════════════════════════════════
# DONE
# ══════════════════════════════════════════════════════════════════════════════

print("\n✓ Routing analysis complete.")
print("  Output files:")
print(f"    {OUTPUT_DIR}/routing_comparison.csv  — per-trial results")
print(f"    {OUTPUT_DIR}/routing_summary.png     — runtime & nodes-explored charts")
print(f"    {OUTPUT_DIR}/routing_paths/trial_XX.png  — one map per trial")
