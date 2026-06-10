"""Dedicated PageRank workflow for the TTC stop network."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
TYPE_I_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPT_DIR.parents[2]
OUTPUT_DIR = SCRIPT_DIR

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if str(TYPE_I_DIR) not in sys.path:
    sys.path.insert(0, str(TYPE_I_DIR))

from graphBuilder import build_graph


SAMPLE_ROWS = 500_000
TOP_STOP_COUNT = 5
TOP_HUB_COUNT = 10
TOP_SUBGRAPH_COUNT = 80

MAJOR_HUBS = [
    "Kennedy",
    "Finch",
    "Wilson",
    "Eglinton",
    "Kipling",
    "Bathurst",
    "Leslie",
    "Union",
    "Bloor",
    "Sheppard West",
    "Scarborough Centre",
    "St George",
    "Dufferin",
    "Pioneer Village",
    "Keele",
    "Glencairn",
    "Mount Dennis",
    "Warden",
    "McCowan",
]


def prepare_graph(sample_rows: int = SAMPLE_ROWS):
    """Build the shared TTC graph and attach PageRank-specific edge weights."""
    cwd = Path.cwd()
    try:
        os.chdir(REPO_ROOT)
        G, giant = build_graph(sample_rows=sample_rows)
    finally:
        os.chdir(cwd)

    pr_graph = G.copy()
    for _, _, data in pr_graph.edges(data=True):
        data["pagerank_weight"] = data.get("trip_count", 1)

    return G, giant, pr_graph


def save_latex_table(df: pd.DataFrame, path: Path) -> None:
    """Write a compact LaTeX table for the slide deck/report."""
    def escape_latex(value) -> str:
        text = str(value)
        replacements = {
            "\\": r"\textbackslash{}",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
            "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
        }
        for source, target in replacements.items():
            text = text.replace(source, target)
        return text

    lines = [
        r"\begin{tabular}{" + " ".join("l" for _ in df.columns) + r"}",
        r"\toprule",
        " & ".join(escape_latex(column) for column in df.columns) + r" \\",
        r"\midrule",
    ]

    for row in df.itertuples(index=False):
        rendered = []
        for value in row:
            if isinstance(value, float):
                rendered.append(f"{value:.10f}")
            else:
                rendered.append(escape_latex(value))
        lines.append(" & ".join(rendered) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_stop_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Build the top stop PageRank table."""
    top_stops = (
        metrics_df.sort_values("pagerank", ascending=False)
        .head(TOP_STOP_COUNT)
        .copy()
    )
    top_stops.insert(0, "rank", range(1, len(top_stops) + 1))
    return top_stops[
        ["rank", "stop_id", "name", "pagerank", "degree", "in_degree", "out_degree", "betweenness"]
    ]


def make_station_table(
    G: nx.DiGraph,
    pagerank: dict[int, float],
    degree_centrality: dict[int, float],
    betweenness: dict[int, float],
) -> pd.DataFrame:
    """Aggregate PageRank across major station hubs."""
    station_rows = []
    for hub in MAJOR_HUBS:
        matching_nodes = [
            n
            for n in G.nodes()
            if hub.lower() in str(G.nodes[n].get("name", "")).lower()
        ]
        if not matching_nodes:
            continue

        representative_node = max(matching_nodes, key=lambda n: pagerank[n])
        platform_stop_count = sum(
            1
            for n in matching_nodes
            if "station" in str(G.nodes[n].get("name", "")).lower()
        )

        station_rows.append(
            {
                "station_name": f"{hub} Station",
                "pagerank_score": sum(pagerank[n] for n in matching_nodes),
                "matched_nodes": len(matching_nodes),
                "degree_centrality": max(degree_centrality[n] for n in matching_nodes),
                "closeness": nx.closeness_centrality(G, u=representative_node),
                "betweenness": max(betweenness[n] for n in matching_nodes),
                "platform_stop_count": platform_stop_count,
            }
        )

    station_df = pd.DataFrame(station_rows)
    station_df = (
        station_df.sort_values("pagerank_score", ascending=False)
        .head(TOP_HUB_COUNT)
        .reset_index(drop=True)
    )
    station_df.insert(0, "rank", range(1, len(station_df) + 1))
    return station_df[
        [
            "rank",
            "station_name",
            "pagerank_score",
            "matched_nodes",
            "degree_centrality",
            "closeness",
            "betweenness",
            "platform_stop_count",
        ]
    ]


def plot_pagerank_geo(G: nx.DiGraph, pagerank: dict[int, float], output_path: Path) -> None:
    """Geographic layout sized and colored by PageRank."""
    pos_geo = {
        n: (G.nodes[n]["lon"], G.nodes[n]["lat"])
        for n in G.nodes()
        if "lon" in G.nodes[n] and "lat" in G.nodes[n]
    }

    nodes_with_pos = list(pos_geo.keys())
    pr_values = np.array([pagerank[n] for n in nodes_with_pos])

    fig, ax = plt.subplots(figsize=(14, 12))
    nx.draw_networkx_nodes(
        G,
        pos_geo,
        nodelist=nodes_with_pos,
        node_size=pr_values / pr_values.max() * 200 + 5,
        node_color=pr_values,
        cmap=cm.plasma,
        alpha=0.8,
        ax=ax,
    )
    nx.draw_networkx_edges(
        G,
        pos_geo,
        nodelist=nodes_with_pos,
        edge_color="#aaaaaa",
        alpha=0.15,
        width=0.3,
        arrows=False,
        ax=ax,
    )

    top20_ids = (
        pd.DataFrame({"stop_id": list(G.nodes()), "pagerank": [pagerank[n] for n in G.nodes()]})
        .nlargest(20, "pagerank")["stop_id"]
        .tolist()
    )
    top20_pos = {n: pos_geo[n] for n in top20_ids if n in pos_geo}
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
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_pagerank_hub_subgraph(
    G: nx.DiGraph,
    pagerank: dict[int, float],
    metrics_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Spring-layout hub subgraph colored by PageRank."""
    top_nodes = metrics_df.nlargest(TOP_SUBGRAPH_COUNT, "degree")["stop_id"].tolist()
    H = G.subgraph(top_nodes).copy()
    pos_spring = nx.spring_layout(H, seed=42, k=0.8)

    h_degrees = dict(H.degree())
    h_sizes = [h_degrees[n] * 30 for n in H.nodes()]
    h_colors = [pagerank[n] for n in H.nodes()]

    fig, ax = plt.subplots(figsize=(14, 12))
    nx.draw_networkx_nodes(
        H,
        pos_spring,
        node_size=h_sizes,
        node_color=h_colors,
        cmap=cm.viridis,
        alpha=0.9,
        ax=ax,
    )
    nx.draw_networkx_edges(H, pos_spring, edge_color="#cccccc", alpha=0.5, width=0.8, ax=ax)
    nx.draw_networkx_labels(
        H,
        pos_spring,
        labels={n: G.nodes[n]["name"] for n in H.nodes()},
        font_size=5,
        ax=ax,
    )

    sm = plt.cm.ScalarMappable(
        cmap=cm.viridis,
        norm=plt.Normalize(
            vmin=min(pagerank[n] for n in H.nodes()),
            vmax=max(pagerank[n] for n in H.nodes()),
        ),
    )
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="PageRank")

    ax.set_title(
        f"Top-{TOP_SUBGRAPH_COUNT} TTC Stops by Degree — Spring Layout\n"
        "(node size = degree, colour = PageRank)",
        fontsize=13,
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_rank_barh(df: pd.DataFrame, name_col: str, value_col: str, title: str, output_path: Path, color: str) -> None:
    """Save a compact horizontal bar chart for ranked outputs."""
    fig, ax = plt.subplots(figsize=(12, max(4, len(df) * 0.55)))
    ordered = df.sort_values(value_col, ascending=True)
    ax.barh(ordered[name_col], ordered[value_col], color=color)
    if value_col == "pagerank":
        ax.set_xlabel("PageRank")
    elif value_col == "pagerank_score":
        ax.set_xlabel("PageRank Score")
    else:
        ax.set_xlabel(value_col.replace("_", " ").title())
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    print("=" * 60)
    print("  EECS4414 — TTC PageRank Analysis")
    print("=" * 60)
    print("\nLoading graph …")

    G, _giant, pr_graph = prepare_graph()
    print()

    print("Computing centrality metrics …")
    degree_centrality = nx.degree_centrality(G)
    nx.set_node_attributes(G, degree_centrality, "degree_centrality")

    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    nx.set_node_attributes(G, in_deg, "in_degree")
    nx.set_node_attributes(G, out_deg, "out_degree")

    pagerank = nx.pagerank(pr_graph, weight="pagerank_weight")
    nx.set_node_attributes(G, pagerank, "pagerank")

    print("  (betweenness approximation with k=200 — remove k= for exact result) …")
    betweenness = nx.betweenness_centrality(G, k=200, normalized=True)
    nx.set_node_attributes(G, betweenness, "betweenness")

    metrics_df = pd.DataFrame(
        {
            "stop_id": list(G.nodes()),
            "name": [G.nodes[n].get("name", "") for n in G.nodes()],
            "degree": [G.degree(n) for n in G.nodes()],
            "in_degree": [G.in_degree(n) for n in G.nodes()],
            "out_degree": [G.out_degree(n) for n in G.nodes()],
            "pagerank": [pagerank[n] for n in G.nodes()],
            "betweenness": [betweenness[n] for n in G.nodes()],
        }
    )

    node_metrics_path = OUTPUT_DIR / "ttc_node_metrics.csv"
    metrics_df.to_csv(node_metrics_path, index=False)
    print(f"  → Full metrics saved to {node_metrics_path}")

    stop_top5 = make_stop_table(metrics_df)
    stop_top5_path = OUTPUT_DIR / "ttc_stop_pagerank_top5.csv"
    stop_top5.to_csv(stop_top5_path, index=False)
    save_latex_table(
        stop_top5[["rank", "name", "pagerank"]].rename(
            columns={"name": "Stop Name", "pagerank": "PageRank Score"}
        ),
        OUTPUT_DIR / "ttc_stop_pagerank_top5.tex",
    )
    plot_rank_barh(
        stop_top5,
        "name",
        "pagerank",
        "Top TTC Stops by PageRank",
        OUTPUT_DIR / "ttc_stop_pagerank_top5.png",
        "#4C72B0",
    )

    station_top10 = make_station_table(G, pagerank, degree_centrality, betweenness)
    station_top10_path = OUTPUT_DIR / "historical_hub_pagerank_top10.csv"
    station_top10.to_csv(station_top10_path, index=False)
    save_latex_table(
        station_top10[["rank", "station_name", "pagerank_score"]].rename(
            columns={"station_name": "Station Name", "pagerank_score": "PageRank Score"}
        ),
        OUTPUT_DIR / "historical_hub_pagerank_top10.tex",
    )
    plot_rank_barh(
        station_top10,
        "station_name",
        "pagerank_score",
        "Top TTC Station Hubs by PageRank",
        OUTPUT_DIR / "historical_hub_pagerank_top10.png",
        "#DD8452",
    )

    plot_pagerank_geo(G, pagerank, OUTPUT_DIR / "pagerank_geo_layout.png")
    plot_pagerank_hub_subgraph(G, pagerank, metrics_df, OUTPUT_DIR / "pagerank_hub_subgraph.png")

    print("\n✓ PageRank outputs written")
    print("  Output files:")
    print(f"    {node_metrics_path}  — per-node centrality table")
    print(f"    {stop_top5_path}  — top 5 stops by PageRank")
    print(f"    {OUTPUT_DIR / 'ttc_stop_pagerank_top5.tex'}")
    print(f"    {OUTPUT_DIR / 'ttc_stop_pagerank_top5.png'}")
    print(f"    {station_top10_path}  — top 10 station hubs by PageRank")
    print(f"    {OUTPUT_DIR / 'historical_hub_pagerank_top10.tex'}")
    print(f"    {OUTPUT_DIR / 'historical_hub_pagerank_top10.png'}")
    print(f"    {OUTPUT_DIR / 'pagerank_geo_layout.png'}")
    print(f"    {OUTPUT_DIR / 'pagerank_hub_subgraph.png'}")


if __name__ == "__main__":
    main()
