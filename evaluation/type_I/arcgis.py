import pandas as pd

DATA_DIR = "dataset/completegtfs"

stops = pd.read_csv(f"{DATA_DIR}/stops.txt", low_memory=False)
stop_times = pd.read_csv(f"{DATA_DIR}/stop_times.txt", low_memory=False)
trips = pd.read_csv(f"{DATA_DIR}/trips.txt", low_memory=False)
routes = pd.read_csv(f"{DATA_DIR}/routes.txt", low_memory=False)

# Clean column names
stops.columns = stops.columns.str.strip()
stop_times.columns = stop_times.columns.str.strip()
trips.columns = trips.columns.str.strip()
routes.columns = routes.columns.str.strip()

# Make IDs strings so merges work
stop_times["trip_id"] = stop_times["trip_id"].astype(str)
trips["trip_id"] = trips["trip_id"].astype(str)
trips["route_id"] = trips["route_id"].astype(str)
routes["route_id"] = routes["route_id"].astype(str)

# Merge route info into stop_times
stop_times = stop_times.merge(
    trips[["trip_id", "route_id"]],
    on="trip_id",
    how="left"
)

stop_times = stop_times.merge(
    routes[["route_id", "route_type", "route_short_name"]],
    on="route_id",
    how="left"
)

# Pick one route type per stop
stop_routes = stop_times.groupby("stop_id").first().reset_index()

arcgis_stops = stops.merge(
    stop_routes[["stop_id", "route_type", "route_short_name"]],
    on="stop_id",
    how="left"
)

def get_mode(route_type):
    if route_type == 1:
        return "Subway"
    elif route_type == 0:
        return "Streetcar"
    elif route_type == 3:
        return "Bus"
    else:
        return "Other"

arcgis_stops["mode"] = arcgis_stops["route_type"].apply(get_mode)

arcgis_stops = arcgis_stops.rename(columns={
    "stop_lat": "latitude",
    "stop_lon": "longitude"
})

arcgis_stops.to_csv("ttc_stops_arcgis_colored.csv", index=False)

print("Created ttc_stops_arcgis_colored.csv")
print(arcgis_stops["mode"].value_counts())