"""
EECS4414 — TTC Graph Builder (Shared Module)
=============================================
Builds the directed stop-to-stop TTC network from GTFS data.
Import this in any analysis script to get a consistent graph:

    from graph_builder import build_graph
    G, giant = build_graph()

Parameters
----------
sample_rows : int or None
    Number of rows to read from stop_times.csv.
    Set to None for the full dataset (slower but complete).
    Default: 500_000  (fast, covers most of the network)

Returns
-------
G : nx.DiGraph
    Full directed graph.  Each node has: name, lat, lon.
    Each edge has: weight (1 / trip-frequency count).
                   Lower weight = more trips = cheaper for Dijkstra.
                   Dijkstra therefore favours well-served corridors.
giant : nx.DiGraph
    Subgraph induced by the largest weakly-connected component of G.

Run from the project root (ttc-network-analysis/):
    python evaluation/type_I/graph_builder.py   ← prints a quick summary
"""

import csv as _csv

import pandas as pd
import networkx as nx


# ──────────────────────────────────────────────────────────────────────────────
# Default configuration  (override by passing arguments to build_graph)
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_SAMPLE_ROWS = 500_000


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_stops() -> pd.DataFrame:
    """Load stop coordinates, trying completegtfs/ then Complete GTFS/."""
    for path in (
        "dataset/completegtfs/stops.csv",
        "dataset/Complete GTFS/stops.txt",
    ):
        try:
            df = pd.read_csv(
                path,
                usecols=["stop_id", "stop_name", "stop_lat", "stop_lon"],
            )
            print(f"  [graph_builder] stops  ← {path}")
            return df
        except FileNotFoundError:
            continue
    raise FileNotFoundError(
        "Cannot find stops file in dataset/completegtfs/ or dataset/Complete GTFS/"
    )


def _load_stop_times(sample_rows) -> pd.DataFrame:
    """Load stop_times, selecting only the three columns we need."""
    needed = {"trip_id", "stop_id", "stop_sequence"}

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
            print(f"  [graph_builder] stop_times ← {path}"
                  + (f"  (first {sample_rows:,} rows)" if sample_rows else ""))
            return df
        except FileNotFoundError:
            continue
    raise FileNotFoundError(
        "Cannot find stop_times file in dataset/completegtfs/ or dataset/Complete GTFS/"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def build_graph(sample_rows: int = DEFAULT_SAMPLE_ROWS):
    """
    Build and return the TTC stop network.

    Parameters
    ----------
    sample_rows : int or None
        Rows to read from stop_times.  None = full dataset.

    Returns
    -------
    G     : nx.DiGraph  — full network
    giant : nx.DiGraph  — largest weakly-connected component
    """

    # ── Step 1: stops → nodes ─────────────────────────────────────────────────
    stops = _load_stops()
    stops = stops.dropna(subset=["stop_lat", "stop_lon"])
    stops["stop_id"] = stops["stop_id"].astype(int)
    print(f"  [graph_builder] {len(stops):,} stops loaded")

    # ── Step 2: stop_times → edges ────────────────────────────────────────────
    stop_times = _load_stop_times(sample_rows)
    stop_times = stop_times.sort_values(["trip_id", "stop_sequence"])
    stop_times["next_stop"] = (
        stop_times.groupby("trip_id")["stop_id"].shift(-1)
    )

    edges_df = stop_times.dropna(subset=["next_stop"]).copy()
    edges_df["stop_id"]   = edges_df["stop_id"].astype(int)
    edges_df["next_stop"] = edges_df["next_stop"].astype(int)

    edge_weights = (
        edges_df.groupby(["stop_id", "next_stop"])
        .size()
        .reset_index(name="trip_count")
    )
    # Invert so Dijkstra treats high-frequency edges as cheaper (more desirable)
    edge_weights["weight"] = 1.0 / edge_weights["trip_count"]
    print(f"  [graph_builder] {len(edge_weights):,} unique directed edges")

    # ── Step 3: construct DiGraph ─────────────────────────────────────────────
    G = nx.DiGraph()

    for _, row in stops.iterrows():
        G.add_node(
            row["stop_id"],
            name=row["stop_name"],
            lat=row["stop_lat"],
            lon=row["stop_lon"],
        )

    for _, row in edge_weights.iterrows():
        if row["stop_id"] in G.nodes and row["next_stop"] in G.nodes:
            G.add_edge(
                row["stop_id"],
                row["next_stop"],
                weight=row["weight"],        # 1 / trip_count  (lower = busier)
                trip_count=row["trip_count"], # raw trip frequency for reference
            )

    print(f"  [graph_builder] graph built → "
          f"{G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    # ── Step 4: giant component ───────────────────────────────────────────────
    G_und      = G.to_undirected()
    giant_set  = max(nx.connected_components(G_und), key=len)
    giant      = G.subgraph(giant_set).copy()
    print(f"  [graph_builder] giant component → "
          f"{giant.number_of_nodes():,} nodes, {giant.number_of_edges():,} edges")

    return G, giant


# ──────────────────────────────────────────────────────────────────────────────
# Quick smoke-test when run directly
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Building TTC graph …")
    G, giant = build_graph()
    print("\nDone.")
    print(f"  Full graph  : {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    print(f"  Giant comp. : {giant.number_of_nodes():,} nodes, {giant.number_of_edges():,} edges")