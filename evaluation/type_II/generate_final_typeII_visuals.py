import os
import pandas as pd
import matplotlib.pyplot as plt

BASE = "evaluation/type_II"
OUT = "evaluation/type_II/final_typeII_visuals"

os.makedirs(OUT, exist_ok=True)

print("Saving final Type II visuals to:", OUT)

# -----------------------------
# Updated stop_times metrics
# -----------------------------
structure = pd.DataFrame({
    "Metric": ["Nodes", "Edges", "Average path length"],
    "Value": [8369, 10737, 72.3936]
})

local = pd.DataFrame({
    "Metric": ["Density", "Clustering coefficient"],
    "Value": [0.000306632, 0.054175]
})

hub_betweenness = pd.DataFrame({
    "Station": [
        "Wilson Station",
        "Sheppard West Station",
        "Bloor Station",
        "Dufferin Station",
        "Finch Station",
        "Scarborough Centre Station",
        "Pioneer Village Station",
        "Kennedy Station",
        "Eglinton Station",
        "Kipling Station"
    ],
    "Betweenness": [
        0.069373,
        0.054378,
        0.053692,
        0.052125,
        0.051183,
        0.050599,
        0.048085,
        0.047942,
        0.045802,
        0.043529
    ]
})

# -----------------------------
# 1. Hub betweenness
# -----------------------------
plt.figure(figsize=(12, 6))
plt.bar(hub_betweenness["Station"], hub_betweenness["Betweenness"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("Betweenness Centrality")
plt.title("Top TTC Station Hubs by Betweenness Centrality")
plt.tight_layout()
plt.savefig(f"{OUT}/01_top_station_hubs_betweenness.png", dpi=300)
plt.close()

# -----------------------------
# 2. Structure metrics
# -----------------------------
plt.figure(figsize=(8, 5))
plt.bar(structure["Metric"], structure["Value"])
for i, v in enumerate(structure["Value"]):
    plt.text(i, v + 250, f"{v:.1f}", ha="center")
plt.ylabel("Value")
plt.title("TTC Network Structure Metrics")
plt.tight_layout()
plt.savefig(f"{OUT}/02_network_structure_metrics.png", dpi=300)
plt.close()

# -----------------------------
# 3. Local connectivity metrics
# -----------------------------
plt.figure(figsize=(7, 5))
plt.bar(local["Metric"], local["Value"])
for i, v in enumerate(local["Value"]):
    plt.text(i, v + 0.002, f"{v:.6f}", ha="center")
plt.ylabel("Metric Value")
plt.title("TTC Local Connectivity Metrics")
plt.tight_layout()
plt.savefig(f"{OUT}/03_local_connectivity_metrics.png", dpi=300)
plt.close()

# -----------------------------
# 4. Traffic bottlenecks
# -----------------------------
traffic = pd.read_csv(f"{BASE}/traffic_bottlenecks_2024.csv")

plt.figure(figsize=(12, 6))
plt.bar(traffic["route"], traffic["passengers_per_vehicle_hour"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("Passengers per Vehicle Hour")
plt.title("Top TTC Crowding Bottlenecks in 2024")
plt.tight_layout()
plt.savefig(f"{OUT}/04_traffic_bottlenecks_2024.png", dpi=300)
plt.close()

# -----------------------------
# 5. Bus delays
# -----------------------------
bus = pd.read_csv(f"{BASE}/disruption_bus_top10.csv")

plt.figure(figsize=(10, 6))
plt.bar(bus["Route"].astype(str), bus["total_delay_min"])
plt.xlabel("Bus Route")
plt.ylabel("Total Delay Minutes")
plt.title("Top Bus Routes by Total Delay Minutes")
plt.tight_layout()
plt.savefig(f"{OUT}/05_bus_disruption_bottlenecks.png", dpi=300)
plt.close()

# -----------------------------
# 6. Streetcar delays
# -----------------------------
streetcar = pd.read_csv(f"{BASE}/disruption_streetcar_top10.csv")

plt.figure(figsize=(10, 6))
plt.bar(streetcar["Route"].astype(str), streetcar["total_delay_min"])
plt.xlabel("Streetcar Route")
plt.ylabel("Total Delay Minutes")
plt.title("Top Streetcar Routes by Total Delay Minutes")
plt.tight_layout()
plt.savefig(f"{OUT}/06_streetcar_disruption_bottlenecks.png", dpi=300)
plt.close()

# -----------------------------
# 7. Subway delay hotspots
# -----------------------------
subway = pd.read_csv(f"{BASE}/disruption_subway_top10.csv")

plt.figure(figsize=(12, 6))
plt.bar(subway["Station"], subway["total_delay_min"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("Total Delay Minutes")
plt.title("Top Subway Stations by Total Delay Minutes")
plt.tight_layout()
plt.savefig(f"{OUT}/07_subway_disruption_hotspots.png", dpi=300)
plt.close()

print("Done. Created final visuals in:")
print(OUT)