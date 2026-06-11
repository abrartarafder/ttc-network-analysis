# type II plots - reads csvs from same folder, saves to plots/

import os

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(OUT, exist_ok=True)

STATION_COLOR = "#003F87"
STREET_COLOR = "#4C72B0"
HUB_COLOR = "#003F87"


def bar_chart(labels, values, title, ylabel, filename, colors=None):
    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(labels))
    if colors is None:
        colors = [STATION_COLOR] * len(labels)
    bars = ax.bar(x, values, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, filename), dpi=300)
    plt.close(fig)


def pool_top10(metric, metric_label, filename):
    from matplotlib.patches import Patch

    pool = pd.read_csv(os.path.join(SCRIPT_DIR, "type_ii_ranking_pool.csv"))
    fig, ax = plt.subplots(figsize=(12, 6))
    top = pool.nlargest(10, metric)
    colors = [
        STATION_COLOR if t == "station" else STREET_COLOR
        for t in top["entity_type"]
    ]
    x = range(len(top))
    bars = ax.bar(x, top[metric], color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(top["display_name"], rotation=45, ha="right")
    ax.set_ylabel(metric_label)
    ax.set_title(f"Top 10 — Stations vs Street Stops ({metric_label})")
    for bar, val in zip(bars, top[metric]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.legend(
        handles=[
            Patch(facecolor=STATION_COLOR, label="Collapsed station"),
            Patch(facecolor=STREET_COLOR, label="Street stop"),
        ],
        loc="upper right",
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, filename), dpi=300)
    plt.close(fig)


def hub_vs_street_pool():
    # 19 hubs vs random street stops for comparison charts
    hubs = pd.read_csv(os.path.join(SCRIPT_DIR, "type_ii_major_hubs_collapsed.csv"))
    pool = pd.read_csv(os.path.join(SCRIPT_DIR, "type_ii_ranking_pool.csv"))
    streets = pool[pool["entity_type"] == "street_stop"].copy()

    hub_rows = []
    for _, hub in hubs.iterrows():
        hub_rows.append(
            {
                "display_name": hub["hub_display_name"],
                "entity_type": "major_hub",
                "Betweenness": hub["Betweenness"],
                "Closeness": hub["Closeness"],
                "Degree": hub["Degree"],
                "route_count": hub["route_count"],
            }
        )
    return pd.concat([pd.DataFrame(hub_rows), streets], ignore_index=True)


def comparison_chart(
    pool,
    metric,
    metric_label,
    title,
    filename,
    n=20,
    hub_color=HUB_COLOR,
    other_color=STREET_COLOR,
    other_label="Individual stop",
    value_fmt="{:.4f}",
):
    from matplotlib.patches import Patch

    top = pool.nlargest(n, metric).copy()
    colors = [
        hub_color if t == "major_hub" else other_color for t in top["entity_type"]
    ]

    fig, ax = plt.subplots(figsize=(14, 7))
    x = range(len(top))
    bars = ax.bar(x, top[metric], color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(top["display_name"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(metric_label)
    ax.set_title(title)
    for bar, val in zip(bars, top[metric]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            value_fmt.format(val),
            ha="center",
            va="bottom",
            fontsize=7,
        )
    ax.legend(
        handles=[
            Patch(facecolor=hub_color, label="Major hub (19)"),
            Patch(facecolor=other_color, label=other_label),
        ],
        loc="upper right",
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, filename), dpi=300)
    plt.close(fig)


def hub_chart(metric, col, title, filename, ascending=False):
    hubs = pd.read_csv(os.path.join(SCRIPT_DIR, "type_ii_hub_bottleneck_combined.csv"))
    hubs = hubs.sort_values(col, ascending=ascending, na_position="last").head(10)
    bar_chart(
        hubs["hub_name"].tolist(),
        hubs[col].tolist(),
        title,
        metric,
        filename,
        colors=[HUB_COLOR] * len(hubs),
    )


def main():
    # run typeii.py first or csvs wont exist
    pool_top10("Betweenness", "Betweenness Centrality", "P1_stops_top10_betweenness.png")

    hub_chart(
        "Betweenness Centrality",
        "betweenness",
        "Top 10 Major Hubs — Betweenness Centrality",
        "H1_hub_betweenness.png",
    )
    hub_chart(
        "Total Delay (minutes)",
        "total_delay_min_all_sources",
        "Top 10 Major Hubs — Total Delay (all sources)",
        "H4_hub_total_delay.png",
    )
    hubs = pd.read_csv(os.path.join(SCRIPT_DIR, "type_ii_major_hubs_collapsed.csv"))
    top_routes = hubs.nlargest(10, "route_count")
    bar_chart(
        top_routes["stop_name"].tolist(),
        top_routes["route_count"].tolist(),
        "Top 10 Major Hubs — Route Count",
        "Route Count",
        "H8_hub_route_count.png",
        colors=[HUB_COLOR] * len(top_routes),
    )

    hub_street = hub_vs_street_pool()
    for metric, label, fmt in [
        ("Betweenness", "Betweenness Centrality", "{:.4f}"),
        ("Closeness", "Closeness Centrality", "{:.4f}"),
        ("Degree", "Degree Centrality", "{:.4f}"),
        ("route_count", "Route Count", "{:.0f}"),
    ]:
        tag = metric.lower().replace(" ", "_")
        comparison_chart(
            hub_street,
            metric,
            label,
            f"Top 20 — Major Hubs vs Street Stops ({label})",
            f"C1_hub_vs_street_{tag}.png",
            n=20,
            other_label="Street stop",
            value_fmt=fmt,
        )

    # N1 N2 - basic graph stats
    summary = pd.read_csv(os.path.join(SCRIPT_DIR, "type_ii_network_summary.csv"))
    summary["value"] = pd.to_numeric(summary["value"], errors="coerce")
    struct = summary[summary["metric"].isin(["nodes", "edges", "average_path_length"])]
    local = summary[summary["metric"].isin(["density", "clustering_coefficient"])]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(struct["metric"], struct["value"])
    for i, v in enumerate(struct["value"]):
        ax.text(i, float(v), f"{float(v):.2f}", ha="center", va="bottom")
    ax.set_title("TTC Network Structure Metrics (Type II)")
    ax.set_ylabel("Value")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "N1_network_structure.png"), dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(local["metric"], local["value"])
    for i, v in enumerate(local["value"]):
        ax.text(i, float(v), f"{float(v):.6f}", ha="center", va="bottom", fontsize=8)
    ax.set_title("Density and Clustering Coefficient")
    ax.set_ylabel("Value")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "N2_density_clustering.png"), dpi=300)
    plt.close(fig)

    # degree dist - most stops are degree 2
    stops = pd.read_csv(os.path.join(SCRIPT_DIR, "type_ii_stops_metrics_full.csv"))
    degrees = stops["graph_degree"].astype(int)
    degrees = degrees[degrees > 0]

    freq = degrees.value_counts().sort_index()
    k_vals = freq.index.to_numpy()
    n_vals = freq.values.astype(float)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(k_vals, n_vals, color="black", linewidth=2.5, solid_capstyle="round")
    ax.set_xlabel("Number of Links", fontsize=12)
    ax.set_ylabel("Number of Nodes", fontsize=12)
    ax.set_title(
        "TTC Stop Network — Degree Distribution\n"
        "(Type II, 464-trip undirected skeleton)",
        fontsize=11,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "N3_degree_distribution.png"), dpi=300)
    plt.close(fig)

    # log-log one goes in the report
    mask = (k_vals > 0) & (n_vals > 0)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(
        k_vals[mask],
        n_vals[mask],
        color="black",
        linewidth=2.5,
        marker="o",
        markersize=4,
        solid_capstyle="round",
    )
    ax.set_xlabel("Number of Links (log)", fontsize=12)
    ax.set_ylabel("Number of Nodes (log)", fontsize=12)
    ax.set_title(
        "TTC Stop Network — Degree Distribution (log-log)\n"
        "(Type II, 464-trip undirected skeleton)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "N3_degree_distribution_loglog.png"), dpi=300)
    plt.close(fig)

    print("done, plots in", OUT)


if __name__ == "__main__":
    main()
