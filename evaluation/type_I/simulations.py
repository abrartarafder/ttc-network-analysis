"""
Simple TTC Disruption Simulation
================================
1. Builds the TTC graph using graphBuilder.py.
2. Picks one source and destination stop.
3. Runs Dijkstra and A* with no disruption.
4. Creates simple disruptions:
   - delays: increase edge weights
   - station closure: remove stops
   - route interruption: remove edges
5. Runs Dijkstra and A* again after each disruption.
6. Creates a graph comparing no disruption vs disruption.

Run from the project root:
    python3 evaluation/type_I/simulations.py

Outputs:
    outputs/simulation_results.csv
    outputs/simulation_baseline_comparison.png
    outputs/simulation_maps/*.png
"""

import math
import os
import random
import sys
import time

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(__file__))
from graphBuilder import build_graph


OUTPUT_DIR = "outputs"
MAPS_DIR = f"{OUTPUT_DIR}/simulation_maps"
RANDOM_SEED = 42
DELAY_MULTIPLIER = 4
NUM_DELAYED_STOPS = 5
NUM_CLOSED_STOPS = 1
NUM_INTERRUPTED_EDGES = 4

random.seed(RANDOM_SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MAPS_DIR, exist_ok=True)


def haversine_km(lat1, lon1, lat2, lon2):
    """Return straight-line distance between two latitude/longitude points."""
    radius_km = 6371
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)

    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(d_lon / 2) ** 2
    )
    return radius_km * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def astar_heuristic(graph, target):
    """
    A* needs a heuristic.
    Here we use straight-line geographic distance to the destination.
    """
    def heuristic(node, goal):
        try:
            node_data = graph.nodes[node]
            target_data = graph.nodes[target]
            return haversine_km(
                node_data["lat"],
                node_data["lon"],
                target_data["lat"],
                target_data["lon"],
            )
        except KeyError:
            return 0

    return heuristic


def make_weight_counter():
    """
    Count how many edges the algorithm checks.
    This gives us the 'search work' value.
    """
    counter = {"edges_checked": 0}

    def weight_function(u, v, edge_data):
        counter["edges_checked"] += 1
        return edge_data.get("weight", 1)

    return weight_function, counter


def calculate_path_cost(graph, path):
    """Add up all edge weights in a path."""
    total = 0
    for i in range(len(path) - 1):
        current_stop = path[i]
        next_stop = path[i + 1]
        total += graph[current_stop][next_stop].get("weight", 1)
    return total


def run_dijkstra(graph, source, target):
    """Run Dijkstra and return useful result information."""
    weight_function, counter = make_weight_counter()

    start_time = time.perf_counter()
    path = nx.dijkstra_path(graph, source, target, weight=weight_function)
    runtime = time.perf_counter() - start_time

    return {
        "algorithm": "Dijkstra",
        "status": "path_found",
        "path": path,
        "path_cost": calculate_path_cost(graph, path),
        "hops": len(path) - 1,
        "edges_evaluated": counter["edges_checked"],
        "runtime_s": runtime,
    }


def run_astar(graph, source, target):
    """Run A* and return useful result information."""
    weight_function, counter = make_weight_counter()

    start_time = time.perf_counter()
    path = nx.astar_path(
        graph,
        source,
        target,
        heuristic=astar_heuristic(graph, target),
        weight=weight_function,
    )
    runtime = time.perf_counter() - start_time

    return {
        "algorithm": "A*",
        "status": "path_found",
        "path": path,
        "path_cost": calculate_path_cost(graph, path),
        "hops": len(path) - 1,
        "edges_evaluated": counter["edges_checked"],
        "runtime_s": runtime,
    }


def run_both_algorithms(graph, source, target):
    """Run Dijkstra and A* on the same graph."""
    results = []

    for runner in (run_dijkstra, run_astar):
        try:
            results.append(runner(graph, source, target))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            results.append({
                "algorithm": "Dijkstra" if runner == run_dijkstra else "A*",
                "status": "no_path",
                "path": [],
                "path_cost": np.nan,
                "hops": np.nan,
                "edges_evaluated": np.nan,
                "runtime_s": np.nan,
            })

    return results


def choose_route(graph):
    """Pick one source and destination that have a route between them."""
    nodes = list(graph.nodes())

    while True:
        source, target = random.sample(nodes, 2)
        try:
            path = nx.dijkstra_path(graph, source, target, weight="weight")
            if len(path) > 20:
                return source, target, path
        except nx.NetworkXNoPath:
            pass


def get_important_stops_on_route(graph, path, count):
    """
    Choose important stops from the selected route.
    Important means the stop has many connections.
    """
    middle_stops = path[1:-1]
    ranked_stops = sorted(middle_stops, key=lambda stop: graph.degree(stop), reverse=True)
    return ranked_stops[:count]


def stop_names(graph, stops):
    """Convert stop ids into readable stop names."""
    return [graph.nodes[stop].get("name", str(stop)) for stop in stops]


def edge_names(graph, edges):
    """Convert route edges into readable stop-to-stop names."""
    readable_edges = []
    for start, end in edges:
        start_name = graph.nodes[start].get("name", str(start))
        end_name = graph.nodes[end].get("name", str(end))
        readable_edges.append(f"{start_name} -> {end_name}")
    return readable_edges


def create_delay_scenario(graph, baseline_path):
    """
    Delay scenario:
    Increase edge weights around a few important stops.
    """
    delayed_graph = graph.copy()
    delayed_stops = get_important_stops_on_route(graph, baseline_path, NUM_DELAYED_STOPS)

    for u, v, edge_data in delayed_graph.edges(data=True):
        if u in delayed_stops or v in delayed_stops:
            edge_data["weight"] = edge_data.get("weight", 1) * DELAY_MULTIPLIER

    reason = (
        "Simulated service delay: edge weights near selected stops were increased "
        f"by {DELAY_MULTIPLIER}x."
    )
    return delayed_graph, "delays", reason, stop_names(graph, delayed_stops), []


def create_station_closure_scenario(graph, baseline_path):
    """
    Station closure scenario:
    Remove a few important stops from the graph.
    """
    closed_graph = graph.copy()
    closed_stops = get_important_stops_on_route(graph, baseline_path, NUM_CLOSED_STOPS)
    closed_graph.remove_nodes_from(closed_stops)
    reason = "Simulated station closure: selected stops were removed from the graph."
    return closed_graph, "station_closure", reason, stop_names(graph, closed_stops), []


def create_route_interruption_scenario(graph, baseline_path):
    """
    Route interruption scenario:
    Remove a few edges from the middle of the original route.
    """
    interrupted_graph = graph.copy()
    route_edges = list(zip(baseline_path, baseline_path[1:]))

    start = len(route_edges) // 2
    edges_to_remove = route_edges[start:start + NUM_INTERRUPTED_EDGES]
    interrupted_graph.remove_edges_from(edges_to_remove)

    reason = "Simulated route interruption: selected connections were removed from the route."
    return interrupted_graph, "route_interruption", reason, [], edge_names(graph, edges_to_remove)


def add_results(
    rows,
    scenario_name,
    source,
    target,
    results,
    reason="No disruption.",
    affected_stops=None,
    affected_edges=None,
):
    """Add algorithm results to the output table."""
    affected_stops = affected_stops or []
    affected_edges = affected_edges or []

    for result in results:
        rows.append({
            "scenario": scenario_name,
            "reason": reason,
            "affected_stops": "; ".join(affected_stops),
            "affected_edges": "; ".join(affected_edges),
            "algorithm": result["algorithm"],
            "status": result["status"],
            "path_cost": result["path_cost"],
            "hops": result["hops"],
            "edges_evaluated": result["edges_evaluated"],
            "runtime_s": result["runtime_s"],
            "source_id": source,
            "target_id": target,
            "path": result["path"],
        })


def get_positions(graph):
    """Get map positions from stop longitude and latitude."""
    return {
        node: (graph.nodes[node]["lon"], graph.nodes[node]["lat"])
        for node in graph.nodes()
        if "lon" in graph.nodes[node] and "lat" in graph.nodes[node]
    }


def draw_route(ax, graph, positions, path, color, label, width):
    """Draw a route path on the map."""
    if not path:
        return

    path_edges = [
        (path[i], path[i + 1])
        for i in range(len(path) - 1)
        if path[i] in positions and path[i + 1] in positions
    ]
    path_nodes = [node for node in path if node in positions]

    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=path_edges,
        edge_color=color,
        width=width,
        alpha=0.85,
        arrows=True,
        arrowsize=8,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=path_nodes,
        node_color=color,
        node_size=18,
        alpha=0.9,
        ax=ax,
        label=label,
    )


def stop_label(graph, stop, label):
    """Create a map label with stop name/address and coordinates."""
    stop_data = graph.nodes[stop]
    name = stop_data.get("name", stop)
    lat = stop_data.get("lat", 0)
    lon = stop_data.get("lon", 0)
    return f"{label}\n{name}\nlat: {lat:.5f}, lon: {lon:.5f}"


def plot_route_map(
    graph,
    scenario_name,
    algorithm,
    source,
    target,
    baseline_path,
    disruption_path,
    reason,
    affected_stops,
    affected_edges,
):
    """Make one simple route map for one algorithm and one disruption scenario."""
    positions = get_positions(graph)

    fig, ax = plt.subplots(figsize=(12, 10))

    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=list(positions.keys()),
        node_size=2,
        node_color="#cccccc",
        alpha=0.35,
        ax=ax,
    )

    draw_route(ax, graph, positions, baseline_path, "#4C72B0", "No disruption", 3)
    route_color = "#DD8452" if algorithm == "Dijkstra" else "#55A868"
    draw_route(ax, graph, positions, disruption_path, route_color, f"{algorithm} disruption", 2.5)

    for stop, label, color in (
        (source, "SOURCE", "green"),
        (target, "TARGET", "red"),
    ):
        if stop in positions:
            ax.scatter(
                positions[stop][0],
                positions[stop][1],
                s=120,
                color=color,
                edgecolors="white",
                linewidths=1,
                zorder=5,
            )
            ax.annotate(
                stop_label(graph, stop, label),
                xy=positions[stop],
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=7,
                fontweight="bold",
                color=color,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.75),
            )

    affected_text = ""
    if affected_stops:
        affected_text = "Affected stops: " + "; ".join(affected_stops[:3])
    elif affected_edges:
        affected_text = "Affected edges: " + "; ".join(affected_edges[:2])

    ax.set_title(
        f"TTC Route Map: {algorithm} - {scenario_name.replace('_', ' ').title()}\n"
        f"{reason}\n{affected_text}",
        fontsize=10,
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="lower right")
    plt.tight_layout()

    algorithm_file = "astar" if algorithm == "A*" else "dijkstra"
    output_path = f"{MAPS_DIR}/{scenario_name}_{algorithm_file}.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved {output_path}")


def plot_scenario_maps(
    graph,
    scenario_name,
    source,
    target,
    baseline_results,
    disruption_results,
    reason,
    affected_stops,
    affected_edges,
):
    """Create separate Dijkstra and A* maps for one disruption scenario."""
    baseline_path = next(
        result["path"]
        for result in baseline_results
        if result["algorithm"] == "Dijkstra" and result["status"] == "path_found"
    )

    for result in disruption_results:
        plot_route_map(
            graph,
            scenario_name,
            result["algorithm"],
            source,
            target,
            baseline_path,
            result["path"],
            reason,
            affected_stops,
            affected_edges,
        )


def plot_algorithm_comparison(results_df):
    """
    Create the main comparison image.

    For every disruption scenario, it compares:
    - Dijkstra with no disruption
    - Dijkstra with disruption
    - A* with no disruption
    - A* with disruption
    """
    scenarios = [
        scenario
        for scenario in results_df["scenario"].unique()
        if scenario != "no_disruption"
    ]

    metrics = [
        ("path_cost", "Path Cost", "Total path cost"),
        ("hops", "Route Length", "Hops"),
        ("edges_evaluated", "Search Work", "Edges evaluated"),
    ]

    x = np.arange(len(scenarios))
    bar_width = 0.18

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle(
        "Dijkstra and A* - No Disruption vs Disruption",
        fontsize=15,
        fontweight="bold",
    )

    bars = [
        ("Dijkstra", "no_disruption", -1.5 * bar_width, "#4C72B0", "Dijkstra no disruption"),
        ("Dijkstra", "disruption", -0.5 * bar_width, "#DD8452", "Dijkstra disruption"),
        ("A*", "no_disruption", 0.5 * bar_width, "#55A868", "A* no disruption"),
        ("A*", "disruption", 1.5 * bar_width, "#C44E52", "A* disruption"),
    ]

    for ax, (column, title, ylabel) in zip(axes, metrics):
        for algorithm, condition, offset, color, label in bars:
            values = []
            for scenario in scenarios:
                lookup_scenario = "no_disruption" if condition == "no_disruption" else scenario
                row = results_df[
                    (results_df["scenario"] == lookup_scenario)
                    & (results_df["algorithm"] == algorithm)
                    & (results_df["status"] == "path_found")
                ]
                values.append(row.iloc[0][column] if not row.empty else np.nan)

            ax.bar(x + offset, values, bar_width, color=color, label=label)

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([name.replace("_", "\n") for name in scenarios])
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    output_path = f"{OUTPUT_DIR}/simulation_baseline_comparison.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved {output_path}")


def main():
    print("=" * 60)
    print("Simple TTC Disruption Simulation")
    print("=" * 60)

    print("\nLoading graph...")
    _, graph = build_graph()

    source, target, baseline_path = choose_route(graph)
    print(f"\nSelected source: {graph.nodes[source].get('name', source)}")
    print(f"Selected target: {graph.nodes[target].get('name', target)}")
    print(f"Original route length: {len(baseline_path) - 1} hops")

    rows = []

    print("\nRunning no-disruption case...")
    no_disruption_results = run_both_algorithms(graph, source, target)
    add_results(rows, "no_disruption", source, target, no_disruption_results)

    scenarios = [
        create_delay_scenario(graph, baseline_path),
        create_station_closure_scenario(graph, baseline_path),
        create_route_interruption_scenario(graph, baseline_path),
    ]

    print("\nRunning disruption cases...")
    for disrupted_graph, scenario_name, reason, affected_stops, affected_edges in scenarios:
        print(f"  {scenario_name}")
        print(f"    reason: {reason}")
        if affected_stops:
            print("    affected stops:")
            for stop in affected_stops:
                print(f"      - {stop}")
        if affected_edges:
            print("    affected edges:")
            for edge in affected_edges:
                print(f"      - {edge}")

        disruption_results = run_both_algorithms(disrupted_graph, source, target)
        add_results(
            rows,
            scenario_name,
            source,
            target,
            disruption_results,
            reason,
            affected_stops,
            affected_edges,
        )
        plot_scenario_maps(
            graph,
            scenario_name,
            source,
            target,
            no_disruption_results,
            disruption_results,
            reason,
            affected_stops,
            affected_edges,
        )

    results_df = pd.DataFrame(rows)
    results_df["path_cost"] = results_df["path_cost"].round(2)
    results_df["runtime_s"] = results_df["runtime_s"].round(6)

    csv_path = f"{OUTPUT_DIR}/simulation_results.csv"
    results_df.drop(columns=["path"]).to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")

    plot_algorithm_comparison(results_df)

    print("\nDone.")
    print("Main output:")
    print(f"  {OUTPUT_DIR}/simulation_baseline_comparison.png")
    print(f"  {MAPS_DIR}/delays_dijkstra.png")
    print(f"  {MAPS_DIR}/delays_astar.png")
    print(f"  {MAPS_DIR}/station_closure_dijkstra.png")
    print(f"  {MAPS_DIR}/station_closure_astar.png")
    print(f"  {MAPS_DIR}/route_interruption_dijkstra.png")
    print(f"  {MAPS_DIR}/route_interruption_astar.png")


if __name__ == "__main__":
    main()
