"""
EECS4414 — TTC Edge-Weight Comparison
======================================
Compares how different edge-weight definitions change routing behaviour
on the TTC stop network.  The same Dijkstra shortest-path algorithm is
applied under three weight schemes so that only the cost definition
changes — isolating the effect of the weight method on the chosen path.

Three edge-weight methods
--------------------------
1. Trip Frequency  — weight = number of trips using that stop-to-stop leg.
                     Higher weight = busier connection.  Dijkstra favours
                     high-frequency (well-served) corridors.

2. Travel Time     — weight = scheduled travel time in seconds between
                     consecutive stops (from departure_time columns).
                     Minimises actual journey duration.

3. Hop Count       — weight = 1 for every edge (unweighted).
                     Minimises the number of stops regardless of frequency
                     or time.  Equivalent to BFS shortest path.

For each weight method, Dijkstra is run on NUM_TRIALS random
source→destination pairs (same pairs across all methods so results are
directly comparable).  The script records path cost, hops, and runtime,
then produces:
  - a summary table printed to the console
  - outputs/weight_comparison.csv       — per-trial results
  - outputs/weight_summary.png          — grouped bar charts
  - outputs/weight_paths/trial_XX.png   — one map per trial showing all
                                          three paths overlaid

Run from the project root (ttc-network-analysis/):
  python evaluation/type_I/type1_weights.py
"""

import csv as _csv
import math
import os
import random
import time
import warnings

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

# ── Shared graph ──────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, os.path.dirname(__file__))
from graphBuilder import build_graph

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

NUM_TRIALS   = 10
RANDOM_SEED  = 42       # same seed as type1_routing.py → same stop pairs
SAMPLE_ROWS  = 500_000
OUTPUT_DIR   = "outputs"
PATHS_DIR    = f"{OUTPUT_DIR}/weight_paths"

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PATHS_DIR,  exist_ok=True)

print("=" * 62)
print("  EECS4414 — TTC Edge-Weight Comparison")
print("=" * 62)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — BASE GRAPH  (trip-frequency weights, from graphBuilder)
# ══════════════════════════════════════════════════════════════════════════════

print("\nStep 1 — Building base graph …")
G, giant = build_graph()
giant_nodes = list(giant.nodes())
print()


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 1 — INITIAL TTC GRAPH STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

print("─" * 54)
print("  Table 1: Initial TTC Graph Statistics")
print("─" * 54)
t1_col = [26, 16, 14]
t1_fmt = "  " + "  ".join(f"{{:<{w}}}" for w in t1_col)
print(t1_fmt.format("Graph Type", "Number of Nodes", "Number of Edges"))
print("  " + "─" * (sum(t1_col) + 2 * len(t1_col)))
print(t1_fmt.format(
    "Stop-Level Graph",
    f"{G.number_of_nodes():,}",
    f"{G.number_of_edges():,}",
))
print(t1_fmt.format(
    "Weighted Graph (Giant Comp.)",
    f"{giant.number_of_nodes():,}",
    f"{giant.number_of_edges():,}",
))
print()


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 2 — NETWORK PROPERTY RESULTS
# ══════════════════════════════════════════════════════════════════════════════
# Computed on the giant component (the largest connected subgraph).
#
# Graph Density              — fraction of possible edges that exist.
# Average Clustering Coeff.  — how often a stop's neighbours are also connected
#                              to each other.  Computed on undirected view.
# Average Shortest Path Len. — mean hops across sampled node pairs (k=300
#                              sample used; exact would be too slow).
# Connected Components       — weakly connected components in the full graph G.
# ─────────────────────────────────────────────────────────────────────────────

print("Computing Table 2 network properties …")

density = nx.density(giant)

giant_und = giant.to_undirected()
avg_clustering = nx.average_clustering(giant_und)

print("  (sampling shortest path length — may take a moment) …")
sample_nodes_t2 = random.sample(giant_nodes, min(300, len(giant_nodes)))
path_lengths = []
for src in sample_nodes_t2:
    lengths = nx.single_source_shortest_path_length(giant, src)
    path_lengths.extend(l for n, l in lengths.items() if n != src)
avg_spl = sum(path_lengths) / len(path_lengths) if path_lengths else float("nan")

n_components = nx.number_weakly_connected_components(G)

print()
print("─" * 50)
print("  Table 2: Network Property Results")
print("─" * 50)
t2_col = [34, 14]
t2_fmt = "  " + "  ".join(f"{{:<{w}}}" for w in t2_col)
print(t2_fmt.format("Metric", "Value"))
print("  " + "─" * (sum(t2_col) + 2 * len(t2_col)))
print(t2_fmt.format("Graph Density",                  f"{density:.6f}"))
print(t2_fmt.format("Average Clustering Coefficient", f"{avg_clustering:.4f}"))
print(t2_fmt.format("Average Shortest Path Length",   f"{avg_spl:.2f} hops"))
print(t2_fmt.format("Number of Connected Components", f"{n_components:,}"))
print()
print("  (Properties computed on giant component unless noted)")
print()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — BUILD TRAVEL-TIME WEIGHTS
# ══════════════════════════════════════════════════════════════════════════════
# Parse departure_time from stop_times to compute seconds between consecutive
# stops on the same trip.  Where multiple trips share the same edge we take
# the median travel time (robust to outliers from express / overnight trips).
# ──────────────────────────────────────────────────────────────────────────────

print("Step 2 — Computing travel-time edge weights …")


def time_to_seconds(t: str) -> int:
    """Convert 'HH:MM:SS' to total seconds.  Handles hour > 23 (overnight)."""
    try:
        h, m, s = t.strip().split(":")
        return int(h) * 3600 + int(m) * 60 + int(s)
    except Exception:
        return None


def load_stop_times_with_time(sample_rows):
    needed = {"trip_id", "stop_id", "stop_sequence", "departure_time"}

    def _read(path):
        return pd.read_csv(
            path,
            usecols=lambda c: c.strip().strip('"') in needed,
            nrows=sample_rows,
            encoding="utf-8-sig",
            quoting=_csv.QUOTE_NONE,
        )

    for path in (
        "dataset/completegtfs/stop_times.csv",
        "dataset/Complete GTFS/stop_times.csv",
    ):
        try:
            df = _read(path)
            df.columns = df.columns.str.strip().str.strip('"')
            df["trip_id"] = df["trip_id"].astype(str).str.strip('"')
            print(f"  [weights] stop_times ← {path}")
            return df
        except FileNotFoundError:
            continue
    raise FileNotFoundError("Cannot find stop_times file.")


st = load_stop_times_with_time(SAMPLE_ROWS)
st = st.sort_values(["trip_id", "stop_sequence"])
st["dep_sec"]      = st["departure_time"].apply(time_to_seconds)
st["next_stop"]    = st.groupby("trip_id")["stop_id"].shift(-1)
st["next_dep_sec"] = st.groupby("trip_id")["dep_sec"].shift(-1)

st = st.dropna(subset=["next_stop", "dep_sec", "next_dep_sec"])
st["travel_sec"] = st["next_dep_sec"] - st["dep_sec"]

# Drop negative/zero times (data artefacts at trip boundaries)
st = st[st["travel_sec"] > 0]
st["stop_id"]   = st["stop_id"].astype(int)
st["next_stop"] = st["next_stop"].astype(int)

# Median travel time per directed edge
time_weights = (
    st.groupby(["stop_id", "next_stop"])["travel_sec"]
    .median()
    .reset_index(name="travel_time")
)
print(f"  [weights] {len(time_weights):,} edges with travel-time data")

# Build a lookup dict: (u, v) → median travel seconds
time_weight_map = {
    (int(r.stop_id), int(r.next_stop)): r.travel_time
    for r in time_weights.itertuples()
}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — ATTACH ALL WEIGHTS TO THE GIANT COMPONENT
# ══════════════════════════════════════════════════════════════════════════════
# We add travel_time and hop attributes alongside the existing weight
# (trip frequency) so a single graph object supports all three schemes.
# ──────────────────────────────────────────────────────────────────────────────

print("\nStep 3 — Attaching weight attributes to giant component …")

missing_time = 0
for u, v, data in giant.edges(data=True):
    # Travel time (fall back to median network travel time if missing)
    tt = time_weight_map.get((u, v))
    if tt is None or tt <= 0:
        missing_time += 1
        tt = 90   # ~90 s fallback ≈ median TTC stop-to-stop time
    data["travel_time"] = tt
    # Hop count — always 1
    data["hop"] = 1

pct_missing = 100 * missing_time / giant.number_of_edges()
print(f"  Travel-time fallback applied to {missing_time:,} edges "
      f"({pct_missing:.1f}% of {giant.number_of_edges():,})")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — WEIGHT SCHEMES
# ══════════════════════════════════════════════════════════════════════════════

WEIGHT_SCHEMES = {
    "Trip Frequency": "weight",       # existing attr from graphBuilder
    "Travel Time":    "travel_time",  # seconds between stops
    "Hop Count":      "hop",          # always 1
}

SCHEME_COLORS = {
    "Trip Frequency": "#4C72B0",
    "Travel Time":    "#DD8452",
    "Hop Count":      "#55A868",
}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — SAMPLE TRIAL PAIRS  (same seed → same pairs as type1_routing.py)
# ══════════════════════════════════════════════════════════════════════════════

print(f"\nStep 4 — Sampling {NUM_TRIALS} reachable trial pairs …")

trial_pairs, attempts = [], 0
while len(trial_pairs) < NUM_TRIALS and attempts < NUM_TRIALS * 20:
    attempts += 1
    src, dst = random.sample(giant_nodes, 2)
    if nx.has_path(giant, src, dst):
        trial_pairs.append((src, dst))

print(f"  {len(trial_pairs)} pairs selected\n")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — RUN DIJKSTRA UNDER EACH WEIGHT SCHEME
# ══════════════════════════════════════════════════════════════════════════════

print("Step 5 — Running Dijkstra under each weight scheme …\n")

header = (
    f"{'#':<4} {'Weight Method':<18} {'Runtime (s)':<13} "
    f"{'Path Cost':<13} {'Hops':<6} "
    f"{'Source':<28} {'→  Destination'}"
)
print(header)
print("─" * len(header))

results   = []
all_paths = {i: {} for i in range(1, len(trial_pairs) + 1)}

for i, (src, dst) in enumerate(trial_pairs, 1):
    src_name = giant.nodes[src].get("name", str(src))[:26]
    dst_name = giant.nodes[dst].get("name", str(dst))[:26]

    for scheme_name, weight_attr in WEIGHT_SCHEMES.items():
        try:
            t0      = time.perf_counter()
            path    = nx.dijkstra_path(giant, src, dst, weight=weight_attr)
            elapsed = time.perf_counter() - t0

            cost = nx.dijkstra_path_length(giant, src, dst, weight=weight_attr)
            hops = len(path) - 1

            all_paths[i][scheme_name] = path

            print(
                f"{i:<4} {scheme_name:<18} {elapsed:<13.6f} "
                f"{cost:<13.1f} {hops:<6} {src_name:<28}    {dst_name}"
            )
            results.append({
                "trial":        i,
                "source_id":    src,
                "source_name":  src_name,
                "dest_id":      dst,
                "dest_name":    dst_name,
                "weight_method": scheme_name,
                "runtime_s":    round(elapsed, 6),
                "path_cost":    round(cost, 2),
                "hops":         hops,
            })

        except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
            print(f"  Trial {i} / {scheme_name} skipped: {e}")

    print()   # blank line between trials

# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — SAVE CSV
# ══════════════════════════════════════════════════════════════════════════════

results_df = pd.DataFrame(results)
csv_path   = f"{OUTPUT_DIR}/weight_comparison.csv"
results_df.to_csv(csv_path, index=False)
print(f"  → Results saved to {csv_path}\n")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════

summary = (
    results_df
    .groupby("weight_method")
    .agg(
        avg_runtime_s  =("runtime_s",  "mean"),
        avg_path_cost  =("path_cost",  "mean"),
        avg_hops       =("hops",       "mean"),
    )
    .reindex(list(WEIGHT_SCHEMES.keys()))   # consistent order
    .reset_index()
)

print("═" * 62)
print("  SUMMARY — Dijkstra Under Different Edge-Weight Methods")
print("═" * 62)
print(f"\n  Trials : {NUM_TRIALS}    |    "
      f"Graph : {giant.number_of_nodes():,} nodes, "
      f"{giant.number_of_edges():,} edges\n")

col_w   = [18, 16, 14, 10]
headers = ["Weight Method", "Avg Runtime (s)", "Avg Path Cost", "Avg Hops"]
row_fmt = "  " + "  ".join(f"{{:<{w}}}" for w in col_w)

print(row_fmt.format(*headers))
print("  " + "─" * (sum(col_w) + 2 * len(col_w)))
for _, row in summary.iterrows():
    print(row_fmt.format(
        row["weight_method"],
        f"{row['avg_runtime_s']:.6f}",
        f"{row['avg_path_cost']:.1f}",
        f"{row['avg_hops']:.1f}",
    ))
print()

# ── Path agreement: how often do methods choose the same route? ───────────────
print("  Path agreement across weight methods (same hops = same path length):")
for i in range(1, len(trial_pairs) + 1):
    trial_rows = results_df[results_df["trial"] == i]
    hops = trial_rows.set_index("weight_method")["hops"].to_dict()
    agree = len(set(hops.values())) == 1
    tag   = "✓ agree" if agree else "✗ differ"
    hop_str = "  |  ".join(f"{m}: {h} hops" for m, h in hops.items())
    print(f"    Trial {i:2d}: {tag}   [{hop_str}]")
print()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 9 — VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════

print("Step 6 — Generating plots …")

# ── 9a. Grouped bar charts: hops and runtime per trial ────────────────────────
schemes = list(WEIGHT_SCHEMES.keys())
n_schemes = len(schemes)
x = np.arange(len(trial_pairs))
bar_w = 0.25

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("Dijkstra Under Different Edge-Weight Methods — TTC Network",
             fontsize=13, fontweight="bold")

for idx, scheme in enumerate(schemes):
    sub  = results_df[results_df["weight_method"] == scheme]
    offs = (idx - 1) * bar_w   # centre the group around each trial tick

    axes[0].bar(x + offs, sub["hops"].values,      bar_w,
                label=scheme, color=SCHEME_COLORS[scheme])
    axes[1].bar(x + offs, sub["runtime_s"].values, bar_w,
                label=scheme, color=SCHEME_COLORS[scheme])

for ax, ylabel, title in [
    (axes[0], "Number of Hops",    "Path Length (Hops) per Trial"),
    (axes[1], "Runtime (seconds)", "Runtime per Trial"),
]:
    ax.set_xlabel("Trial")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([f"T{i+1}" for i in range(len(trial_pairs))])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/weight_summary.png", dpi=150)
plt.close()
print(f"  → {OUTPUT_DIR}/weight_summary.png saved")


# ── 9b. One map per trial with all three paths overlaid ──────────────────────
pos_geo = {
    n: (giant.nodes[n]["lon"], giant.nodes[n]["lat"])
    for n in giant.nodes()
    if "lon" in giant.nodes[n] and "lat" in giant.nodes[n]
}


def draw_path_on_ax(ax, path, color, lw=2.5, alpha=0.85):
    path_edges = list(zip(path, path[1:]))
    path_pos   = {n: pos_geo[n] for n in path if n in pos_geo}
    nx.draw_networkx_nodes(
        giant, path_pos, nodelist=list(path_pos.keys()),
        node_size=25, node_color=color, alpha=alpha, ax=ax,
    )
    nx.draw_networkx_edges(
        giant, pos_geo, edgelist=path_edges,
        edge_color=color, width=lw, alpha=alpha,
        arrows=True, arrowsize=8, ax=ax,
    )


for trial_num, (src_id, dst_id) in enumerate(trial_pairs, 1):
    paths = all_paths.get(trial_num, {})
    if not paths:
        continue

    fig, ax = plt.subplots(figsize=(12, 10))

    # Background nodes
    nx.draw_networkx_nodes(
        giant, pos_geo,
        nodelist=list(pos_geo.keys()),
        node_size=2, node_color="#cccccc", alpha=0.35, ax=ax,
    )

    # Draw each scheme's path (thicker for frequency, thinner for others)
    line_widths = {"Trip Frequency": 3.5, "Travel Time": 2.5, "Hop Count": 1.5}
    for scheme, path in paths.items():
        draw_path_on_ax(ax, path, SCHEME_COLORS[scheme],
                        lw=line_widths[scheme])

    # Source / destination labels
    for nid, label, color in [(src_id, "SOURCE", "green"), (dst_id, "DEST", "red")]:
        if nid in pos_geo:
            ax.annotate(
                f"{label}\n{giant.nodes[nid].get('name', nid)[:30]}",
                xy=pos_geo[nid], fontsize=7, color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
            )

    legend_patches = [
        mpatches.Patch(color=SCHEME_COLORS[s], label=s) for s in schemes
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9)

    # Per-scheme hop count in the title
    hop_info = "   |   ".join(
        f"{s}: {paths[s] and len(paths[s])-1} hops"
        for s in schemes if s in paths
    )
    ax.set_title(
        f"Weight Comparison — Trial {trial_num}\n"
        f"Source: {giant.nodes[src_id].get('name','?')[:38]}  →  "
        f"Dest: {giant.nodes[dst_id].get('name','?')[:38]}\n"
        f"{hop_info}",
        fontsize=9,
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()

    out_path = f"{PATHS_DIR}/trial_{trial_num:02d}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  → {out_path} saved")

print(f"  All {len(trial_pairs)} trial maps saved to {PATHS_DIR}/")


# ══════════════════════════════════════════════════════════════════════════════
# DONE
# ══════════════════════════════════════════════════════════════════════════════

print("\n✓ Weight comparison complete.")
print("  Output files:")
print(f"    {OUTPUT_DIR}/weight_comparison.csv     — per-trial results")
print(f"    {OUTPUT_DIR}/weight_summary.png        — hops & runtime bar charts")
print(f"    {PATHS_DIR}/trial_XX.png               — one map per trial")