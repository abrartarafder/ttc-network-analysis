import os
import math
import pandas as pd
import networkx as nx


DATA = "./dataset"
OUT = "./evaluation/type_II/Thor_Scripts_Results"

os.makedirs(OUT, exist_ok=True)


# Load GTFS data.

print("Loading trips and stops...")

trips = pd.read_csv(
    f"{DATA}/completegtfs/trips.csv",
    low_memory=False
)

stops = pd.read_csv(
    f"{DATA}/completegtfs/stops.csv"
)


# Clean station names and merge bus bays.

major_hubs = [
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


def clean_station(name):
    if pd.isna(name):
        return name

    name = str(name)

    for hub in major_hubs:
        if hub.lower() in name.lower():
            return hub + " Station"

    return name


stops["clean_stop_name"] = stops["stop_name"].apply(clean_station)


# Pick one trip per route/direction.

trips["trip_id"] = trips["trip_id"].astype(str)

# Keep one representative trip per route/direction.
one_trip = trips.drop_duplicates(
    subset=["route_id", "direction_id"],
    keep="first"
)

trip_ids = one_trip["trip_id"].tolist()

print("Using", len(trip_ids), "trips (one per route/direction)")


# Load stop_times.

print("Loading stop_times (this can take a bit)...")

stop_times = pd.read_csv(
    f"{DATA}/completegtfs/stop_times.txt",
    low_memory=False
)

stop_times["trip_id"] = stop_times["trip_id"].astype(str)

stop_times = stop_times[
    stop_times["trip_id"].isin(trip_ids)
]

stop_times = stop_times.sort_values(
    ["trip_id", "stop_sequence"]
)

print("Stop times filtered down to", len(stop_times), "rows")


# Build the TTC stop-level graph.
# Nodes are stop IDs; edges connect consecutive stops on a trip.

G = nx.Graph()

for trip_id, group in stop_times.groupby("trip_id"):
    stop_list = group["stop_id"].tolist()

    for i in range(len(stop_list) - 1):
        G.add_edge(
            stop_list[i],
            stop_list[i + 1]
        )

print("\nGraph built from stop_times")
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())


# Add walking transfers at station hubs.

print("\nAdding walking transfers at station hubs...")

MAX_WALK = 150  # metres

# Station-like GTFS stops get transfer edges between nearby stop IDs.
hub_stops = stops[
    stops["stop_name"].str.contains(
        "Station",
        case=False,
        na=False
    )
]

hub_stops = hub_stops.reset_index(drop=True)

print("Hub stops found (Station in name):", len(hub_stops))


def get_distance_m(lat1, lon1, lat2, lon2):
    R = 6371000  # Earth radius in metres.

    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)

    dlat = lat2 - lat1
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    return R * 2 * math.asin(math.sqrt(a))


transfer_edges = 0

for i in range(len(hub_stops)):
    for j in range(i + 1, len(hub_stops)):
        stop_a = hub_stops.iloc[i]
        stop_b = hub_stops.iloc[j]

        if stop_a["stop_id"] not in G.nodes():
            continue

        if stop_b["stop_id"] not in G.nodes():
            continue

        dist = get_distance_m(
            stop_a["stop_lat"],
            stop_a["stop_lon"],
            stop_b["stop_lat"],
            stop_b["stop_lon"]
        )

        if dist <= MAX_WALK:
            G.add_edge(
                stop_a["stop_id"],
                stop_b["stop_id"]
            )

            transfer_edges += 1

print("Hub transfer edges added:", transfer_edges)
print("Connected components:", len(list(nx.connected_components(G))))
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())


# Centrality calculations.

print("\nCalculating degree...")
degree = nx.degree_centrality(G)

print("Sampling nodes...")
sample_nodes = list(G.nodes())[:1000]

print("Calculating sampled closeness...")

closeness = {}

for n in sample_nodes:
    closeness[n] = nx.closeness_centrality(
        G,
        u=n
    )

print("Calculating sampled betweenness...")

betweenness = nx.betweenness_centrality(
    G,
    k=20,
    seed=42
)


# Graph-level metrics.

print("\nCalculating graph metrics...")

density = nx.density(G)
clustering = nx.average_clustering(G)

connected_components = list(nx.connected_components(G))

largest_component = max(
    connected_components,
    key=len
)

largest_graph = G.subgraph(largest_component).copy()

sample_nodes_path = list(largest_graph.nodes())[:1000]

sample_graph = largest_graph.subgraph(sample_nodes_path).copy()

sample_components = list(nx.connected_components(sample_graph))

largest_sample_component = max(
    sample_components,
    key=len
)

path_graph = sample_graph.subgraph(largest_sample_component).copy()

avg_path = nx.average_shortest_path_length(path_graph)


# Build the centrality results table.

results = pd.DataFrame({
    "stop_id": list(closeness.keys()),
    "Degree": [degree[n] for n in closeness.keys()],
    "Closeness": [closeness[n] for n in closeness.keys()],
    "Betweenness": [betweenness.get(n, 0) for n in closeness.keys()],
})

results = results.merge(
    stops[
        [
            "stop_id",
            "clean_stop_name",
            "stop_lat",
            "stop_lon"
        ]
    ],
    on="stop_id",
    how="left"
)

results.rename(
    columns={
        "clean_stop_name": "stop_name"
    },
    inplace=True
)

results["is_hub"] = results["stop_name"].isin(
    [x + " Station" for x in major_hubs]
)

top_degree = results.sort_values(
    "Degree",
    ascending=False
).head(10)

top_closeness = results.sort_values(
    "Closeness",
    ascending=False
).head(10)

top_betweenness = results.sort_values(
    "Betweenness",
    ascending=False
).head(10)


# Hub-only centrality tables.
# Collapse repeated station names into one row per station.

hub_results = results[
    results["is_hub"]
].copy()

hub_results_clean = (
    hub_results
    .groupby("stop_name", as_index=False)
    .agg({
        "Degree": "max",
        "Closeness": "max",
        "Betweenness": "max",
        "stop_lat": "mean",
        "stop_lon": "mean"
    })
)

top_hub_degree = (
    hub_results_clean
    .sort_values("Degree", ascending=False)
    .head(10)
)

top_hub_closeness = (
    hub_results_clean
    .sort_values("Closeness", ascending=False)
    .head(10)
)

top_hub_betweenness = (
    hub_results_clean
    .sort_values("Betweenness", ascending=False)
    .head(10)
)


# Network summary.

network_summary = pd.DataFrame({
    "Metric": [
        "Nodes",
        "Edges",
        "Hub transfer edges added",
        "Hub stops used for transfer links",
        "Connected components",
        "Largest component (%)",
        "Density",
        "Clustering coefficient",
        "Average path length (stop hops)",
        "Trips used",
    ],
    "Value": [
        G.number_of_nodes(),
        G.number_of_edges(),
        transfer_edges,
        len(hub_stops),
        len(connected_components),
        100 * len(largest_component) / G.number_of_nodes(),
        density,
        clustering,
        avg_path,
        len(trip_ids),
    ],
})

print("\nNetwork summary:")
print(network_summary)

print("\nTop hub betweenness:")
print(top_hub_betweenness)


# Traffic analysis.

print("\nTraffic...")

pvh = pd.read_csv(
    f"{DATA}/traffic/ttc_passengers_vehicle_hour_2019_2024.csv"
)

pvh["2024"] = pd.to_numeric(
    pvh["2024"],
    errors="coerce"
)

busy_routes = (
    pvh.dropna(subset=["2024"])
    .nlargest(10, "2024")[["Route", "2024"]]
    .rename(
        columns={
            "Route": "route",
            "2024": "passengers_per_vehicle_hour"
        }
    )
)

print(busy_routes)


# Disruption analysis.

print("\nDisruptions...")

bus = pd.read_csv(
    f"{DATA}/disruptions/busses/bus_data.csv"
)

streetcar = pd.read_csv(
    f"{DATA}/disruptions/streetcars/streetcar_data.csv"
)

subway = pd.read_csv(
    f"{DATA}/disruptions/subway/subway_data.csv"
)

bus_delays = (
    bus.groupby("Route")["Min Delay"]
    .sum()
    .reset_index(name="total_delay_min")
    .sort_values("total_delay_min", ascending=False)
    .head(10)
)

sc_delays = (
    streetcar.groupby("Route")["Min Delay"]
    .sum()
    .reset_index(name="total_delay_min")
    .sort_values("total_delay_min", ascending=False)
    .head(10)
)

sub_delays = (
    subway.groupby("Station")["Min Delay"]
    .sum()
    .reset_index(name="total_delay_min")
    .sort_values("total_delay_min", ascending=False)
    .head(10)
)

print(bus_delays)
print(sc_delays)
print(sub_delays)


# Helper for markdown tables.

def df_to_md(df):
    lines = [
        "| " + " | ".join(str(c) for c in df.columns) + " |"
    ]

    lines.append(
        "| " + " | ".join("---" for _ in df.columns) + " |"
    )

    for row in df.itertuples(index=False):
        lines.append(
            "| " + " | ".join(str(v) for v in row) + " |"
        )

    return "\n".join(lines)


# Write the markdown summary.

summary_path = f"{OUT}/TYPE_II_STOPTIMES_RESULTS_SUMMARY.md"

with open(summary_path, "w", encoding="utf-8") as f:
    f.write("# Type II Results Summary (stop_times + station hubs)\n\n")

    f.write("## How the network was built\n\n")

    f.write("1. **Route edges** — consecutive stops from `stop_times.txt` were connected using one representative trip per route/direction.\n")

    f.write("2. **Hub transfers** — major TTC stations were treated as transfer areas. ")

    f.write("Nearby station-related stops within **150 m** were connected with walking transfer edges using haversine distance.\n")

    f.write(f"3. **{transfer_edges} hub transfer edges** were added across **{len(hub_stops)} station-related stops**.\n\n")

    f.write("## 1. Network summary — connectivity and graph structure\n\n")

    f.write("| Metric | Meaning |\n| --- | --- |\n")

    f.write("| Connected components | How many separate pieces the graph breaks into |\n")

    f.write("| Largest component (%) | Share of stops in the main connected network |\n")

    f.write("| Density | How many edges exist compared with all possible edges |\n")

    f.write("| Clustering coefficient | How much stops form small tightly connected groups |\n")

    f.write("| Average path length | Average stop hops in a sample, not travel minutes |\n\n")

    f.write(df_to_md(network_summary) + "\n\n")

    f.write("## 2. Centrality — all stops\n\n")

    f.write("- **Degree** — how many neighbouring stops a stop connects to.\n")

    f.write("- **Closeness** — how efficiently a stop can reach the rest of the network.\n")

    f.write("- **Betweenness** — how often a stop lies on shortest paths, making it a possible bottleneck.\n\n")

    f.write("Betweenness uses sampled `k=20`. Closeness is computed on a 1000-node sample for speed. All metrics are computed on the graph after hub transfer edges are added.\n\n")

    f.write("### Top 10 stops by degree\n\n")
    f.write(df_to_md(top_degree.drop(columns=["is_hub"])) + "\n\n")

    f.write("### Top 10 stops by closeness\n\n")
    f.write(df_to_md(top_closeness.drop(columns=["is_hub"])) + "\n\n")

    f.write("### Top 10 stops by betweenness\n\n")
    f.write(df_to_md(top_betweenness.drop(columns=["is_hub"])) + "\n\n")

    f.write("## 3. Centrality at major station hubs\n\n")

    f.write("These tables group repeated station labels into one row per station. ")

    f.write("This avoids showing separate bus bays or platforms as different hubs.\n\n")

    f.write("### Top 10 hub stops by degree\n\n")
    f.write(df_to_md(top_hub_degree) + "\n\n")

    f.write("### Top 10 hub stops by closeness\n\n")
    f.write(df_to_md(top_hub_closeness) + "\n\n")

    f.write("### Top 10 hub stops by betweenness\n\n")
    f.write(df_to_md(top_hub_betweenness) + "\n\n")

    f.write("## 4. Bottlenecks — operational crowding and delays\n\n")

    f.write("Structural bottlenecks come from high betweenness in the graph. ")

    f.write("Operational bottlenecks come from crowding and delay data.\n\n")

    f.write("### Traffic — crowding, 2024 passengers per vehicle hour\n\n")
    f.write(df_to_md(busy_routes) + "\n\n")

    f.write("### Disruptions — bus\n\n")
    f.write(df_to_md(bus_delays) + "\n\n")

    f.write("### Disruptions — streetcar\n\n")
    f.write(df_to_md(sc_delays) + "\n\n")

    f.write("### Disruptions — subway\n\n")
    f.write(df_to_md(sub_delays) + "\n\n")

    f.write("## 5. Routing efficiency and accessibility\n\n")

    f.write(f"- **Routing efficiency:** average path length is about **{round(avg_path, 1)} stop hops** on the largest connected sample. ")

    f.write("This is a structural hop count, not actual TTC travel time.\n")

    f.write(f"- **Connectivity:** the graph has **{len(connected_components)}** connected components, with **{round(100 * len(largest_component) / G.number_of_nodes(), 1)}%** of stops in the largest component. ")

    f.write("This suggests the TTC stop network is mostly connected after station transfer links are added.\n")

    f.write("- **Accessibility:** crowded routes like **504 King** and **64 Main**, delayed bus/streetcar routes such as **41** and **501**, and subway stations such as **Kennedy BD** show where access may be difficult in practice despite network connectivity.\n")


print(f"\nWrote {summary_path}")
print("Done")
