"""
EECS4414 — TTC Transit Network: Metrics & Visualisation
=========================================================
Computes centrality metrics and produces static visualisations for the
TTC stop network.  Graph construction is delegated to graph_builder.py
so the same graph object can be shared with type1_routing.py.

  - networkx   (centrality metrics)
  - pandas     (data wrangling + CSV output)
  - matplotlib (static visualisations)

Expected folder structure (run from project root ttc-network-analysis/):
  dataset/
    completegtfs/   OR   Complete GTFS/
      stops.csv / stops.txt
      stop_times.csv
    disruptions/
      subway/subway_data.csv

Run:  python evaluation/type_I/type1_abrar.py
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import networkx as nx
import os

matplotlib.use("Agg")   # headless — swap to "TkAgg" for interactive use

# ── Shared graph ──────────────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from graphBuilder import build_graph

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("  EECS4414 — TTC Network Metrics & Visualisation")
print("=" * 60)
print("\nLoading graph …")
G, giant = build_graph()
print()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — COMPUTE NETWORK METRICS
# ═══════════════════════════════════════════════════════════════════════════════
#
# degree_centrality   — fraction of other nodes a node is connected to.
#                       High degree = major interchange stop.
#
# betweenness_centrality — how often a node lies on shortest paths.
#                          High betweenness = "bottleneck" stop.
#                          Approximated with k=200 samples for speed;
#                          remove k= for the exact result in the final report.
#
# PageRank            — importance based on what important nodes link to.
#                       Useful for identifying transit hubs.
# ─────────────────────────────────────────────────────────────────────────────

print("Step 4 — Computing centrality metrics …")

# Degree centrality (fast)
deg_cent = nx.degree_centrality(G)
nx.set_node_attributes(G, deg_cent, "degree_centrality")

# In-degree and out-degree (directional breakdown)
in_deg  = dict(G.in_degree())
out_deg = dict(G.out_degree())
nx.set_node_attributes(G, in_deg,  "in_degree")
nx.set_node_attributes(G, out_deg, "out_degree")

# PageRank (fast, direction-aware hub importance)
pagerank = nx.pagerank(G, weight="weight")
nx.set_node_attributes(G, pagerank, "pagerank")

# Betweenness centrality — approximate for speed
print("  (betweenness approximation with k=200 — remove k= for exact result) …")
betweenness = nx.betweenness_centrality(G, k=200, normalized=True)
nx.set_node_attributes(G, betweenness, "betweenness")

# ── Summary table ─────────────────────────────────────────────────────────────
metrics_df = pd.DataFrame({
    "stop_id":     list(G.nodes()),
    "name":        [G.nodes[n].get("name", "") for n in G.nodes()],
    "degree":      [G.degree(n)    for n in G.nodes()],
    "in_degree":   [G.in_degree(n) for n in G.nodes()],
    "out_degree":  [G.out_degree(n) for n in G.nodes()],
    "pagerank":    [pagerank[n]    for n in G.nodes()],
    "betweenness": [betweenness[n] for n in G.nodes()],
})

top10 = metrics_df.sort_values("pagerank", ascending=False).head(10)
print("\n  Top 10 stops by PageRank:")
print(top10[["name", "degree", "pagerank", "betweenness"]].to_string(index=False))

csv_path = f"{OUTPUT_DIR}/ttc_node_metrics.csv"
metrics_df.to_csv(csv_path, index=False)
print(f"\n  → Full metrics saved to {csv_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — VISUALISE
# ═══════════════════════════════════════════════════════════════════════════════
#
#   5a. Geographic layout — stops at real lat/lon, sized by PageRank.
#   5b. Degree distribution — linear + log-log histograms.
#   5c. Top-N hub subgraph — spring layout of the highest-degree stops.
# ─────────────────────────────────────────────────────────────────────────────

print("\nStep 5 — Plotting …")

# ── 5a. Geographic layout ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 12))

pos_geo = {
    n: (G.nodes[n]["lon"], G.nodes[n]["lat"])
    for n in G.nodes()
    if "lon" in G.nodes[n] and "lat" in G.nodes[n]
}

nodes_with_pos = list(pos_geo.keys())
pr_values = np.array([pagerank[n] for n in nodes_with_pos])

node_sizes  = pr_values / pr_values.max() * 200 + 5
node_colors = pr_values

nx.draw_networkx_nodes(
    G, pos_geo,
    nodelist=nodes_with_pos,
    node_size=node_sizes,
    node_color=node_colors,
    cmap=cm.plasma,
    alpha=0.8,
    ax=ax,
)
nx.draw_networkx_edges(
    G, pos_geo,
    nodelist=nodes_with_pos,
    edge_color="#aaaaaa",
    alpha=0.15,
    width=0.3,
    arrows=False,
    ax=ax,
)

# Label only the top-20 hubs
top20_ids    = metrics_df.nlargest(20, "pagerank")["stop_id"].tolist()
top20_pos    = {n: pos_geo[n] for n in top20_ids if n in pos_geo}
top20_labels = {n: G.nodes[n]["name"] for n in top20_pos}
nx.draw_networkx_labels(G, top20_pos, top20_labels, font_size=6, font_color="white", ax=ax)

sm = plt.cm.ScalarMappable(
    cmap=cm.plasma,
    norm=plt.Normalize(vmin=pr_values.min(), vmax=pr_values.max()),
)
sm.set_array([])
plt.colorbar(sm, ax=ax, label="PageRank")

ax.set_title("TTC Stop Network — Geographic Layout\n(node size & colour = PageRank)", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/ttc_geo_layout.png", dpi=150)
plt.close()
print(f"  → {OUTPUT_DIR}/ttc_geo_layout.png saved")


# ── 5b. Degree distribution ───────────────────────────────────────────────────
degrees = [d for _, d in G.degree() if d > 0]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(degrees, bins=30, color="#4C72B0", edgecolor="white")
axes[0].set_xlabel("Degree")
axes[0].set_ylabel("Count")
axes[0].set_title("Degree Distribution (linear)")

log_bins = np.logspace(np.log10(max(1, min(degrees))), np.log10(max(degrees)), 20)
axes[1].hist(degrees, bins=log_bins, color="#DD8452", edgecolor="white")
axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].set_xlabel("Degree (log)")
axes[1].set_ylabel("Count (log)")
axes[1].set_title("Degree Distribution (log-log)")

plt.suptitle("TTC Network — Degree Distribution", fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/ttc_degree_dist.png", dpi=150)
plt.close()
print(f"  → {OUTPUT_DIR}/ttc_degree_dist.png saved")


# ── 5c. Top-hub subgraph ──────────────────────────────────────────────────────
TOP_N = 80   # increase for final report

top_nodes  = metrics_df.nlargest(TOP_N, "degree")["stop_id"].tolist()
H          = G.subgraph(top_nodes).copy()
pos_spring = nx.spring_layout(H, seed=42, k=0.8)

h_degrees = dict(H.degree())
h_sizes   = [h_degrees[n] * 30 for n in H.nodes()]
h_colors  = [pagerank[n] for n in H.nodes()]

fig, ax = plt.subplots(figsize=(14, 12))
nx.draw_networkx_nodes(H, pos_spring, node_size=h_sizes, node_color=h_colors,
                       cmap=cm.viridis, alpha=0.9, ax=ax)
nx.draw_networkx_edges(H, pos_spring, edge_color="#cccccc", alpha=0.5, width=0.8, ax=ax)
nx.draw_networkx_labels(H, pos_spring,
                        labels={n: G.nodes[n]["name"] for n in H.nodes()},
                        font_size=5, ax=ax)

sm2 = plt.cm.ScalarMappable(
    cmap=cm.viridis,
    norm=plt.Normalize(
        vmin=min(pagerank[n] for n in H.nodes()),
        vmax=max(pagerank[n] for n in H.nodes()),
    ),
)
sm2.set_array([])
plt.colorbar(sm2, ax=ax, label="PageRank")

ax.set_title(f"Top-{TOP_N} TTC Stops by Degree — Spring Layout\n"
             "(node size = degree, colour = PageRank)", fontsize=13)
ax.axis("off")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/ttc_hub_subgraph.png", dpi=150)
plt.close()
print(f"  → {OUTPUT_DIR}/ttc_hub_subgraph.png saved")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6 — CONNECTED COMPONENTS & ROBUSTNESS
# ═══════════════════════════════════════════════════════════════════════════════

print("\nStep 6 — Connected components …")

G_und      = G.to_undirected()
components = list(nx.connected_components(G_und))
print(f"  Weakly connected components : {len(components)}")
print(f"  Largest component size      : {max(len(c) for c in components)}")
print(f"  Isolated nodes              : {sum(1 for c in components if len(c) == 1)}")
print(f"  Giant component             : {giant.number_of_nodes():,} nodes, "
      f"{giant.number_of_edges():,} edges")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7 — ENRICH WITH DISRUPTION DATA
# ═══════════════════════════════════════════════════════════════════════════════

print("\nStep 7 — Loading subway disruptions …")

try:
    disruptions = pd.read_csv("dataset/disruptions/subway/subway_data.csv")

    delay_counts = (
        disruptions.groupby("Station")["Min Delay"]
        .agg(total_delay="sum", incidents="count")
        .reset_index()
    )

    stop_name_map = {
        G.nodes[n]["name"].upper(): n
        for n in G.nodes()
        if "name" in G.nodes[n]
    }

    matched = 0
    for _, row in delay_counts.iterrows():
        key = str(row["Station"]).upper()
        if key in stop_name_map:
            nid = stop_name_map[key]
            G.nodes[nid]["total_delay"]  = row["total_delay"]
            G.nodes[nid]["disruptions"]  = row["incidents"]
            matched += 1

    print(f"  → Disruption data attached to {matched} matching nodes")
    print("  Tip: for better matching, consider fuzzy string matching (thefuzz library)")

except FileNotFoundError:
    print("  (Skipping — dataset/disruptions/subway/subway_data.csv not found)")
except Exception as e:
    print(f"  (Skipping disruption enrichment — {e})")


# ═══════════════════════════════════════════════════════════════════════════════
# DONE
# ═══════════════════════════════════════════════════════════════════════════════

print("\n✓ All steps complete.")
print("  Output files:")
print(f"    {OUTPUT_DIR}/ttc_node_metrics.csv  — per-node centrality table")
print(f"    {OUTPUT_DIR}/ttc_geo_layout.png    — geographic stop map")
print(f"    {OUTPUT_DIR}/ttc_degree_dist.png   — degree distribution plots")
print(f"    {OUTPUT_DIR}/ttc_hub_subgraph.png  — top-hub spring-layout subgraph")