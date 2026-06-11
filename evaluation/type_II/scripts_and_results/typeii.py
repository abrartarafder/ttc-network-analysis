# type II - undirected graph, station clusters, centrality + delay tables
# run from repo root (needs dataset/completegtfs)

import math
import os
import re

import networkx as nx
import pandas as pd
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
DATA = os.path.join(REPO_ROOT, "dataset")
OUT = SCRIPT_DIR

os.makedirs(OUT, exist_ok=True)

HUB_RADIUS_M = 150
BETWEENNESS_K_STOP = 500
BETWEENNESS_K_STATION = None  # exact on station graph

STOP_TIMES_LINE = re.compile(
    r'^"?(\d+),(\d{2}:\d{2}:\d{2}),(\d{2}:\d{2}:\d{2}),(\d+),(\d+),'
    r'(.*),(\d),(\d),([\d.]+),(\d)"?$'
)

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

# gtfs names that dont match our hub list exactly
HUB_CLUSTER_OVERRIDES = {
    "McCowan": "McCowan Rt Station",
    "Bloor": "Bloor-Yonge Station",
}

# bloor-yonge was split in gtfs, merge it back
INTERCHANGE_UNION = {
    "Bloor Station": "Bloor-Yonge Station",
    "Yonge Station": "Bloor-Yonge Station",
}

# subway delay csv uses weird station strings
HUB_SUBWAY_DELAY_BASES = {
    "Bloor Station": ["Bloor", "Yonge"],
}


def load_stop_times(data_dir):
    txt_path = os.path.join(data_dir, "completegtfs", "stop_times.txt")
    csv_path = os.path.join(data_dir, "completegtfs", "stop_times.csv")

    if os.path.exists(txt_path):
        df = pd.read_csv(
            txt_path,
            usecols=["trip_id", "stop_id", "stop_sequence"],
            low_memory=False,
        )
        df["trip_id"] = df["trip_id"].astype(str)
        print(f"got stop_times.txt, {len(df):,} rows")
        return df

    rows = []
    with open(csv_path, encoding="utf-8-sig") as handle:
        handle.readline()
        for line in handle:
            line = line.strip()
            if not line:
                continue
            match = STOP_TIMES_LINE.match(line)
            if not match:
                continue
            rows.append((match.group(1), int(match.group(4)), int(match.group(5))))
    df = pd.DataFrame(rows, columns=["trip_id", "stop_id", "stop_sequence"])
    print(f"fallback csv, {len(df):,} rows (might be cut off)")
    return df


def normalize_station_label(label):
    label = re.sub(r"\s+", " ", str(label).strip())
    return re.sub(r"\bstation\b", "Station", label, flags=re.I)


def extract_station_name(name):
    if pd.isna(name):
        return None
    name = str(name).strip()
    if "station" not in name.lower():
        return None

    # dont break sheppard-yonge (learned that the hard way)
    if " - " in name:
        m = re.search(r"-\s*(.+?\s+(?:Rt\s+)?(?:GO\s+)?Station)\s*$", name, re.I)
        if m:
            return normalize_station_label(m.group(1))

    m = re.match(r"^(.+?\s+(?:Rt\s+)?(?:GO\s+)?Station)\b", name, re.I)
    if m:
        return normalize_station_label(m.group(1))

    m = re.search(r"\(([^)]+\s+Station)\)", name, re.I)
    if m:
        return normalize_station_label(m.group(1))

    return None


def apply_interchange_unions(cluster_map):
    for stop_id, label in list(cluster_map.items()):
        unified = INTERCHANGE_UNION.get(label)
        if unified:
            cluster_map[stop_id] = unified
    return cluster_map


def get_distance_m(lat1, lon1, lat2, lon2):
    r = 6371000
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    dlat = lat2 - lat1
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return r * 2 * math.asin(math.sqrt(a))


def assign_station_clusters(stops_df, graph_node_ids, radius_m=HUB_RADIUS_M):
    # name parsing first, then 150m geo for bus bays etc
    graph_nodes = set(graph_node_ids)
    in_graph = stops_df[stops_df["stop_id"].isin(graph_nodes)].copy()

    cluster = {}
    for _, row in in_graph.iterrows():
        label = extract_station_name(row["stop_name"])
        if label:
            cluster[row["stop_id"]] = label

    seeds_by_station = {}
    for _, row in in_graph.iterrows():
        label = extract_station_name(row["stop_name"])
        if label:
            seeds_by_station.setdefault(label, []).append(row["stop_id"])
    for label in seeds_by_station:
        seeds_by_station[label] = list(dict.fromkeys(seeds_by_station[label]))

    seed_coords = {
        sid: (
            in_graph.loc[in_graph["stop_id"] == sid, "stop_lat"].iloc[0],
            in_graph.loc[in_graph["stop_id"] == sid, "stop_lon"].iloc[0],
        )
        for sid in {s for seeds in seeds_by_station.values() for s in seeds}
        if sid in graph_nodes
    }

    for stop_id in [sid for sid in graph_nodes if sid not in cluster]:
        match = in_graph.loc[in_graph["stop_id"] == stop_id]
        if match.empty:
            continue
        row = match.iloc[0]
        best_label = None
        best_dist = float("inf")
        for label, seed_ids in seeds_by_station.items():
            for seed_id in seed_ids:
                if seed_id not in seed_coords:
                    continue
                lat2, lon2 = seed_coords[seed_id]
                dist = get_distance_m(row["stop_lat"], row["stop_lon"], lat2, lon2)
                if dist <= radius_m and dist < best_dist:
                    best_dist = dist
                    best_label = label
        if best_label:
            cluster[stop_id] = best_label

    return cluster


def add_cluster_clique_edges(graph, cluster_map):
    by_cluster = {}
    for node, label in cluster_map.items():
        if node not in graph:
            continue
        by_cluster.setdefault(label, []).append(node)

    added = 0
    for members in by_cluster.values():
        if len(members) < 2:
            continue
        for i, a in enumerate(members):
            for b in members[i + 1 :]:
                if not graph.has_edge(a, b):
                    graph.add_edge(a, b, edge_type="cluster_clique")
                    added += 1
    return added


def add_transfer_edges(graph, stops_df, radius_m=HUB_RADIUS_M):
    # walking links between station stops w/in 150m
    station_stops = stops_df[
        stops_df["stop_name"].astype(str).str.contains("Station", case=False, na=False)
    ].copy()
    station_stops = station_stops[station_stops["stop_id"].isin(graph.nodes())]

    added = 0
    nodes = list(station_stops.itertuples(index=False))
    for i, a in enumerate(nodes):
        for b in nodes[i + 1 :]:
            if graph.has_edge(a.stop_id, b.stop_id):
                continue
            dist = get_distance_m(a.stop_lat, a.stop_lon, b.stop_lat, b.stop_lon)
            if dist <= radius_m:
                graph.add_edge(a.stop_id, b.stop_id, edge_type="walking_transfer")
                added += 1
    return added


def resolve_hub_cluster_row(hub, stations_df):
    if hub in HUB_CLUSTER_OVERRIDES:
        override = HUB_CLUSTER_OVERRIDES[hub]
        match = stations_df[stations_df["stop_name"] == override]
        if len(match):
            return match.iloc[0]

    label = hub + " Station"
    exact = stations_df[stations_df["stop_name"] == label]
    if len(exact):
        return exact.iloc[0]
    return None


def parse_member_stop_ids(member_ids_value):
    if pd.isna(member_ids_value) or member_ids_value == "":
        return []
    return [int(s) for s in str(member_ids_value).split(",") if s.strip()]


def count_routes_in_string(routes_serving):
    if pd.isna(routes_serving) or routes_serving == "":
        return 0
    return len({r.strip() for r in str(routes_serving).split(",") if r.strip()})


def build_crowding_by_gtfs_route(pvh_df):
    # traffic csv route names are messy ("7 BATHURST" etc)
    pvh = pvh_df.copy()
    pvh["2024"] = pd.to_numeric(pvh["2024"], errors="coerce")
    pvh = pvh.dropna(subset=["2024"])

    crowding = {}
    for _, row in pvh.iterrows():
        route_label = str(row["Route"]).strip()
        value = float(row["2024"])
        crowding[route_label] = max(crowding.get(route_label, 0), value)
        num_match = re.match(r"^(\d+)", route_label)
        if num_match:
            num = num_match.group(1)
            crowding[num] = max(crowding.get(num, 0), value)
    return crowding


def match_subway_delays(sub_delays_df, hub_display_name):
    bases = HUB_SUBWAY_DELAY_BASES.get(
        hub_display_name,
        [hub_display_name.replace(" Station", "").strip()],
    )
    station_suffix = (
        r"(STATION|RT\s+STATION|BD\s+STATION|YUS\s+STATION|GO\s+STATION|CTR\s+STATION)\b"
    )
    mask = pd.Series(False, index=sub_delays_df.index)
    for hub_base in bases:
        pattern = rf"^{re.escape(hub_base.upper())}\s+{station_suffix}"
        mask |= sub_delays_df["Station"].astype(str).str.upper().str.match(
            pattern, na=False
        )
    return sub_delays_df[mask]


def routes_for_stop_ids(stop_ids, stop_route_rows):
    mask = stop_route_rows["stop_id"].isin(stop_ids)
    routes = (
        stop_route_rows.loc[mask, "route_short_name"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    return sorted(set(routes)), routes.nunique()


def compute_betweenness(graph, k=None):
    if k is None:
        return nx.betweenness_centrality(graph, normalized=True)
    return nx.betweenness_centrality(graph, k=k, normalized=True, seed=42)


def read_gtfs_table(name):
    for ext in (".txt", ".csv"):
        path = os.path.join(DATA, "completegtfs", name + ext)
        if os.path.exists(path):
            return pd.read_csv(path, low_memory=False)
    raise FileNotFoundError(f"Missing GTFS table: {name}")


def main():
    print("loading gtfs...")
    trips = read_gtfs_table("trips")
    routes = read_gtfs_table("routes")
    routes["route_id"] = routes["route_id"].astype(str)
    stops = read_gtfs_table("stops")
    print("loading stop_times (slow)...")
    stop_times = load_stop_times(DATA)

    trips["direction_id"] = trips["direction_id"].fillna(0).astype(int)
    trips["route_id"] = trips["route_id"].astype(str)
    trips["trip_id"] = trips["trip_id"].astype(str)

    # one trip per route+direction so we dont double count lines
    one_trip_per_route_dir = (
        trips.groupby(["route_id", "direction_id"], as_index=False)
        .first()["trip_id"]
        .tolist()
    )
    print(f"{len(one_trip_per_route_dir)} trips after filter")

    st = stop_times[stop_times["trip_id"].astype(str).isin(one_trip_per_route_dir)].copy()
    st["stop_id"] = st["stop_id"].astype(int)
    st = st.sort_values(["trip_id", "stop_sequence"])

    print("building graph...")
    graph = nx.Graph()
    for trip_id, group in st.groupby("trip_id"):
        seq = group.sort_values("stop_sequence")["stop_id"].tolist()
        for i in range(len(seq) - 1):
            graph.add_edge(seq[i], seq[i + 1], edge_type="route")

    print(f"route edges: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    station_cluster = assign_station_clusters(stops, graph.nodes())
    station_cluster = apply_interchange_unions(station_cluster)
    print(f"{len(set(station_cluster.values()))} station clusters")

    clique_added = add_cluster_clique_edges(graph, station_cluster)
    transfer_added = add_transfer_edges(graph, stops)
    print(f"+{clique_added} clique edges, +{transfer_added} walking")
    print(f"final graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    largest = components[0]
    in_largest = {n: n in largest for n in graph.nodes()}

    print(f"centrality on stops (k={BETWEENNESS_K_STOP})...")
    degree_cent = nx.degree_centrality(graph)
    closeness_cent = nx.closeness_centrality(graph)
    betweenness_cent = compute_betweenness(graph, k=BETWEENNESS_K_STOP)

    results = pd.DataFrame(
        {
            "stop_id": list(graph.nodes()),
            "graph_degree": [graph.degree(n) for n in graph.nodes()],
            "Degree": [degree_cent[n] for n in graph.nodes()],
            "Closeness": [closeness_cent[n] for n in graph.nodes()],
            "Betweenness": [betweenness_cent[n] for n in graph.nodes()],
            "in_largest_component": [in_largest[n] for n in graph.nodes()],
        }
    )
    results = results.merge(
        stops[["stop_id", "stop_name", "stop_lat", "stop_lon"]],
        on="stop_id",
        how="left",
    )
    results["station_cluster"] = results["stop_id"].map(station_cluster)
    results["is_station"] = results["stop_name"].astype(str).str.contains(
        "Station", case=False, na=False
    )

    global stop_route_rows
    stop_route_rows = (
        st.merge(trips[["trip_id", "route_id"]], on="trip_id")
        .merge(routes[["route_id", "route_short_name"]], on="route_id")
        .drop_duplicates(subset=["stop_id", "route_short_name"])
    )
    route_info = results["stop_id"].apply(
        lambda sid: routes_for_stop_ids([sid], stop_route_rows)
    )
    results["route_count"] = route_info.apply(lambda x: x[1])
    results["routes_serving"] = route_info.apply(lambda x: ",".join(x[0]))

    # collapse stations for hub betwenness
    print("station graph...")
    station_graph = nx.Graph()
    cluster_to_stops = {}
    for stop_id, cluster in station_cluster.items():
        if stop_id not in graph:
            continue
        cluster_to_stops.setdefault(cluster, []).append(stop_id)

    for cluster, members in cluster_to_stops.items():
        station_graph.add_node(cluster, entity_type="station", members=members)

    for node in graph.nodes():
        if node in station_cluster:
            continue
        street_node = f"stop:{node}"
        station_graph.add_node(street_node, entity_type="street_stop", members=[node])

    for u, v in graph.edges():
        if u in station_cluster and v in station_cluster:
            if station_cluster[u] == station_cluster[v]:
                continue
            a, b = station_cluster[u], station_cluster[v]
        elif u in station_cluster and v not in station_cluster:
            a, b = station_cluster[u], f"stop:{v}"
        elif v in station_cluster and u not in station_cluster:
            a, b = station_cluster[v], f"stop:{u}"
        else:
            a, b = f"stop:{u}", f"stop:{v}"
        if a != b:
            station_graph.add_edge(a, b)

    print(f"station graph: {station_graph.number_of_nodes()} nodes, {station_graph.number_of_edges()} edges")

    btw_label = "exact" if BETWEENNESS_K_STATION is None else f"k={BETWEENNESS_K_STATION}"
    print(f"station centrality ({btw_label})...")
    st_degree = nx.degree_centrality(station_graph)
    st_closeness = nx.closeness_centrality(station_graph)
    st_betweenness = compute_betweenness(station_graph, k=BETWEENNESS_K_STATION)

    station_rows = []
    for node, attrs in station_graph.nodes(data=True):
        if attrs.get("entity_type") != "station":
            continue
        members = attrs.get("members", [])
        member_df = results[results["stop_id"].isin(members)]
        routes_union = sorted(
            set(
                r.strip()
                for rs in member_df["routes_serving"].dropna()
                for r in str(rs).split(",")
                if r.strip()
            )
        )
        routes_serving = ",".join(routes_union)
        station_rows.append(
            {
                "stop_name": node,
                "entity_type": "station",
                "platform_stop_count": len(members),
                "graph_degree": station_graph.degree(node),
                "Degree": st_degree[node],
                "Closeness": st_closeness[node],
                "Betweenness": st_betweenness[node],
                "route_count": len(routes_union),
                "routes_serving": routes_serving,
                "member_stop_ids": ",".join(str(m) for m in members),
            }
        )

    all_stations_collapsed = pd.DataFrame(station_rows).sort_values(
        "Betweenness", ascending=False
    )
    all_stations_collapsed.to_csv(
        os.path.join(OUT, "type_ii_all_stations_collapsed.csv"), index=False
    )

    # stations vs street stops for charts
    street_pool = results[~results["stop_id"].isin(station_cluster.keys())].copy()
    street_pool["entity_type"] = "street_stop"
    street_pool["display_name"] = street_pool["stop_name"]

    station_pool = all_stations_collapsed.copy()
    station_pool["display_name"] = station_pool["stop_name"]

    ranking_pool = pd.concat(
        [
            station_pool[
                [
                    "display_name",
                    "entity_type",
                    "Degree",
                    "Closeness",
                    "Betweenness",
                    "route_count",
                    "platform_stop_count",
                ]
            ],
            street_pool[
                [
                    "display_name",
                    "entity_type",
                    "Degree",
                    "Closeness",
                    "Betweenness",
                    "route_count",
                ]
            ].assign(platform_stop_count=1),
        ],
        ignore_index=True,
    )
    ranking_pool.to_csv(os.path.join(OUT, "type_ii_ranking_pool.csv"), index=False)

    # 19 hubs for report tables
    hub_rows = []
    major_cluster_names = set()
    for hub in MAJOR_HUBS:
        row = resolve_hub_cluster_row(hub, all_stations_collapsed)
        if row is None:
            print(f"  warn: no cluster for hub {hub}")
            continue
        d = row.to_dict()
        d["hub_display_name"] = hub + " Station"
        d["station_cluster_name"] = row["stop_name"]
        d["stop_name"] = hub + " Station"
        d["entity_type"] = "station_hub"
        major_cluster_names.add(row["stop_name"])
        hub_rows.append(d)

    major_hubs_collapsed = pd.DataFrame(hub_rows)
    if len(major_hubs_collapsed):
        major_hubs_collapsed = major_hubs_collapsed.sort_values(
            "Betweenness", ascending=False
        )
    major_hubs_collapsed.to_csv(
        os.path.join(OUT, "type_ii_major_hubs_collapsed.csv"), index=False
    )

    results["is_major_hub_cluster"] = results["station_cluster"].isin(
        major_cluster_names
    )

    print("joining delays + crowding...")
    pvh = pd.read_csv(
        os.path.join(DATA, "traffic", "ttc_passengers_vehicle_hour_2019_2024.csv")
    )
    crowding_by_route = build_crowding_by_gtfs_route(pvh)

    stop_route_rows["route_key"] = stop_route_rows["route_short_name"].astype(str).str.strip()
    stop_route_rows["pax_per_vehicle_hour_2024"] = stop_route_rows["route_key"].map(
        crowding_by_route
    )
    crowding_per_stop = (
        stop_route_rows.groupby("stop_id")["pax_per_vehicle_hour_2024"]
        .max()
        .reset_index(name="max_pax_per_vehicle_hour_2024_serving_route")
    )
    results = results.merge(crowding_per_stop, on="stop_id", how="left")

    bus = pd.read_csv(os.path.join(DATA, "disruptions", "busses", "bus_data.csv"))
    streetcar = pd.read_csv(
        os.path.join(DATA, "disruptions", "streetcars", "streetcar_data.csv")
    )
    subway = pd.read_csv(os.path.join(DATA, "disruptions", "subway", "subway_data.csv"))

    bus_delays_full = (
        bus.groupby("Route")["Min Delay"].sum().reset_index(name="bus_delay_min")
    )
    bus_delays_full["route_str"] = bus_delays_full["Route"].astype(str).str.strip()
    sc_delays_full = (
        streetcar.groupby("Route")["Min Delay"].sum().reset_index(name="sc_delay_min")
    )
    sc_delays_full["route_str"] = sc_delays_full["Route"].astype(str).str.strip()
    sub_delays_full = (
        subway.groupby("Station")["Min Delay"].sum().reset_index(name="subway_delay_min")
    )

    bottleneck_rows = []
    for _, hub_row in major_hubs_collapsed.iterrows():
        hub_name = hub_row["hub_display_name"]
        stop_ids_for_hub = parse_member_stop_ids(hub_row["member_stop_ids"])

        routes_mask = stop_route_rows["stop_id"].isin(stop_ids_for_hub)
        hub_route_keys = (
            stop_route_rows.loc[routes_mask, "route_short_name"]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        )

        bus_match = bus_delays_full[bus_delays_full["route_str"].isin(hub_route_keys)]
        sc_match = sc_delays_full[sc_delays_full["route_str"].isin(hub_route_keys)]
        sub_match = match_subway_delays(sub_delays_full, hub_name)

        crowding_vals = results.loc[
            results["stop_id"].isin(stop_ids_for_hub),
            "max_pax_per_vehicle_hour_2024_serving_route",
        ]

        bottleneck_rows.append(
            {
                "hub_name": hub_name,
                "station_cluster_name": hub_row["station_cluster_name"],
                "betweenness": round(float(hub_row["Betweenness"]), 6),
                "closeness": round(float(hub_row["Closeness"]), 6),
                "degree_centrality": round(float(hub_row["Degree"]), 6),
                "platform_stop_count": len(stop_ids_for_hub),
                "route_count": count_routes_in_string(hub_row["routes_serving"]),
                "routes_serving_filtered_graph": ",".join(sorted(set(hub_route_keys))),
                "max_pax_per_vehicle_hour_2024": (
                    round(float(crowding_vals.max()), 1) if crowding_vals.notna().any() else ""
                ),
                "bus_delay_min_serving_routes": int(bus_match["bus_delay_min"].sum()),
                "streetcar_delay_min_serving_routes": int(sc_match["sc_delay_min"].sum()),
                "subway_delay_min": int(sub_match["subway_delay_min"].sum()),
                "total_delay_min_all_sources": (
                    int(bus_match["bus_delay_min"].sum())
                    + int(sc_match["sc_delay_min"].sum())
                    + int(sub_match["subway_delay_min"].sum())
                ),
            }
        )

    hub_bottleneck = (
        pd.DataFrame(bottleneck_rows)
        .sort_values("betweenness", ascending=False, na_position="last")
        .reset_index(drop=True)
    )
    hub_bottleneck.to_csv(
        os.path.join(OUT, "type_ii_hub_bottleneck_combined.csv"), index=False
    )

    # route counts
    route_ops = (
        stop_route_rows.groupby("route_short_name")
        .agg(stop_count=("stop_id", "nunique"), trip_count=("trip_id", "nunique"))
        .reset_index()
        .sort_values("stop_count", ascending=False)
    )
    route_ops.to_csv(os.path.join(OUT, "type_ii_routes_operational.csv"), index=False)

    # graph stats for report
    density = nx.density(graph)
    clustering = nx.average_clustering(graph)
    try:
        avg_path = nx.average_shortest_path_length(graph.subgraph(largest))
    except Exception:
        avg_path = float("nan")

    summary = pd.DataFrame(
        {
            "metric": [
                "nodes",
                "edges",
                "route_edges_only",
                "cluster_clique_edges_added",
                "walking_transfer_edges_added",
                "station_clusters",
                "collapsed_station_entities",
                "largest_component_fraction",
                "density",
                "clustering_coefficient",
                "average_path_length",
                "betweenness_k_stop",
                "betweenness_station",
                "trips_used",
            ],
            "value": [
                graph.number_of_nodes(),
                graph.number_of_edges(),
                graph.number_of_edges() - clique_added - transfer_added,
                clique_added,
                transfer_added,
                len(set(station_cluster.values())),
                len(all_stations_collapsed),
                len(largest) / graph.number_of_nodes(),
                density,
                clustering,
                avg_path,
                BETWEENNESS_K_STOP,
                btw_label,
                len(one_trip_per_route_dir),
            ],
        }
    )
    summary.to_csv(os.path.join(OUT, "type_ii_network_summary.csv"), index=False)

    results.to_csv(os.path.join(OUT, "type_ii_stops_metrics_full.csv"), index=False)

    print("\ndone. csvs in", OUT)
    print(summary.to_string(index=False))
    print("\ntop stations:")
    print(all_stations_collapsed.head(10)[["stop_name", "Betweenness", "Closeness", "Degree"]].to_string(index=False))
    print("\ntop major hubs:")
    print(major_hubs_collapsed.head(10)[["stop_name", "Betweenness", "Closeness"]].to_string(index=False))


if __name__ == "__main__":
    main()
