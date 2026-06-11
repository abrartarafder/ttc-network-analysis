# Type II — Complete Findings Reference 

Study guide for slides and report writing. All numbers from `June6_Results/important/` after clustering fixes (Bloor–Yonge merge, Sheppard–Yonge parser fix).

**Pipeline:** `typeii.py` → `type_ii_final_bars.py`  
**Data:** Full GTFS `stop_times.txt` (May 2026 TTC feed); disruption CSVs; 2024 crowding CSV.

---

## 0. What Type II asks (frame this first)

**Type II = network topology and graph properties**, not passenger routing.

| | **Type I** | **Type II** |
|--|-----------|-------------|
| **Question** | How do paths behave under weights & disruptions? | What does the network *look like* structurally? |
| **Graph** | Directed, frequency-weighted, larger sample | Undirected, one trip per route+direction |
| **Weights** | Inverse trip frequency, travel time, etc. | Unweighted (hop count) |
| **Main metrics** | Dijkstra/A*, path cost, disruption rerouting | Degree, density, clustering, betweenness, closeness |
| **Operational data** | Simulated in routing | Joined as **delay** on hub tables (secondary layer) |

**One sentence:** Type I asks “how do you get from A to B when things break?” Type II asks “where is the network thin, hub-heavy, and bridge-like?”

---

## 1. Methodology — what was built

### 1.1 Graph construction pipeline

1. **464 trips** — one trip per `route_id` + `direction_id` from full GTFS `stop_times.txt` (~3.3M rows).
2. **Undirected edges** — consecutive stops on each trip (9,041 route-only edges).
3. **`station_cluster`** — ~**140** collapsed station groups from GTFS names + 150 m geo fallback (~904 stops assigned).
4. **Cluster clique edges** — all stops in the same cluster fully connected (+3,360 edges).
5. **Walking transfer edges** — 150 m links between stops whose names contain “Station” (+71 edges).
6. **Final graph:** **8,056 nodes**, **12,472 edges**, **99.93%** in largest connected component.

### 1.2 Centrality computation

| Level | Graph | Betweenness |
|-------|-------|-------------|
| Stop-level | Full 8k-stop graph | **k=500** sampled (approximate) |
| Station-level (140 clusters) | Collapsed station graph | **Exact** |
| Major hubs (19 names) | Filter on collapsed results | Same as matching cluster row |

Hub charts (H1, C1) use **collapsed station betweenness** (exact). Per-stop CSV uses sampled betweenness.

### 1.3 Major hubs vs station clusters

- **140 `station_cluster` labels** = graph truth (cliques, collapse, all-station rankings).
- **19 `major_hubs`** = presentation filter only (H1, bottleneck table, C1 navy bars).
- Clusters can rank highly without being a “major hub” (e.g. Lawrence, York Mills, Spadina).

### 1.4 Operational joins (delay + crowding)

- Delays: Toronto Open Data bus, streetcar, subway CSVs.
- Matched via **`member_stop_ids`** from each hub’s collapsed cluster (exact match).
- Crowding: 2024 pax/vehicle-hour; routes normalized (`"7 BATHURST"` → `"7"`).
- **18/19 hubs** have crowding data; **Bloor** missing in traffic join.

### 1.5 Clustering fixes

| Issue | Fix | Effect |
|-------|-----|--------|
| `Sheppard-Yonge Station` parsed as `Yonge Station` | Only use `" - "` suffix for street addresses | Sheppard–Yonge separate (~12 stops at ~43.76°N) |
| Bloor (Line 1) + downtown Yonge (Line 2) split | Merged **`Bloor-Yonge Station`** | Real interchange represented |
| St George | Already unified in GTFS | Line 1 + Line 2 in one cluster (5 stops) |

---

## 2. Part 1 — Network property analysis

### 2.1 Global structure metrics

| Metric | Value | Meaning |
|--------|-------|---------|
| **Nodes** | 8,056 | Stops on 464-trip skeleton |
| **Edges** | 12,472 | Route + clique + walking |
| **Density** | 0.000384 | Extremely sparse |
| **Clustering coefficient** | 0.098 | Moderate local grouping at stations |
| **Average path length** | ~30.3 hops | Typical shortest path is long |
| **Largest component** | 99.93% | Well connected overall |
| **Station clusters** | 140 | Collapsed station entities |

**Interpretation:** Sparse **hub-and-spoke** network — long paths, low global density, local cliques at stations.

### 2.2 Degree distribution (plot: `N3_degree_distribution.png`)

| Degree | Stop count | Role |
|--------|------------|------|
| **2** | ~5,960 (~74%) | Middle of a route (prev + next) |
| **1** | 62 (~0.8%) | Terminals / dead ends |
| **0** | 0 | No isolated nodes |
| **Max** | 24 | Hub/terminal tail |

**Key finding:** Sharp peak at degree 2; long tail of high-degree hubs.

**Log-log (`N3_degree_distribution_loglog.png`):** Supports scale-free-like wording. On log scale, degree-1 sits at 10⁰ (one link, not zero).

### 2.3 Density + clustering implications

- **Low density** → few redundant global paths.
- **Moderate clustering** → stations are locally dense despite sparse whole graph.
- **~30 hops** → large geographic scale in unweighted model.

---

## 3. Part 2 — Station hub identification & representation

### 3.1 Collapse logic

- `Bloor Station - Southbound Platform` → `Bloor-Yonge Station` (merged) or other cluster labels.
- Unclustered stops remain **street nodes** (~7,152 in ranking pool).
- **904 stops** have a `station_cluster`.

### 3.2 Why collapse matters

Without clique edges, platforms appear disconnected. Cliques (+3,360 edges) model each station as a **transfer zone**.

### 3.3 Example: Eglinton cluster

- **23** platform/stop members, **11** routes in filtered graph.
- Large clusters absorb nearby bus bays → inflates local connectivity.

### 3.4 The 19 major hubs

Kennedy, Finch, Wilson, Eglinton, Kipling, Bathurst, Leslie, Union, Bloor, Sheppard West, Scarborough Centre, St George, Dufferin, Pioneer Village, Keele, Glencairn, Mount Dennis, Warden, McCowan.

Curated filter for charts — not the only important stations in the graph.

---

## 4. Part 3 — Transportation connectivity & network bottlenecks

**Metric:** **Betweenness** — fraction of shortest paths (hop count) through a node.

### 4.1 Top 15 stations (all 140 — exact betweenness)

| Rank | Station | Betweenness | Closeness | Routes |
|------|---------|-------------|-----------|--------|
| 1 | York Mills | 0.262 | 0.054 | 13 |
| 2 | Lawrence | 0.227 | 0.054 | 8 |
| 3 | Victoria Park | 0.212 | 0.046 | 8 |
| 4 | Eglinton | 0.211 | 0.055 | 11 |
| 5 | Bloor-Yonge | 0.199 | 0.052 | 4 |
| 6 | Warden | 0.185 | 0.045 | 12 |
| 7 | Spadina | 0.183 | 0.052 | 6 |
| 8 | Broadview | 0.181 | 0.050 | 13 |
| 9 | Kennedy | 0.180 | 0.044 | 25 |
| 10 | Cedarvale | 0.175 | 0.053 | 7 |
| 13 | St George | 0.161 | 0.052 | 4 |
| 14 | Sheppard-Yonge | 0.156 | 0.052 | 8 |

**Plot P1:** Top 10 pool entities are all **collapsed stations**, not street stops.

### 4.2 Top 19 major hubs by betweenness (plot: `H1_hub_betweenness.png`)

| Rank | Hub | Betweenness | Routes | Total delay (min) |
|------|-----|-------------|--------|-------------------|
| 1 | Eglinton | 0.211 | 11 | 382,002 |
| 2 | Bloor | 0.199 | 4 | 113,972 |
| 3 | Warden | 0.185 | 12 | 544,049 |
| 4 | Kennedy | 0.180 | 25 | 938,644 |
| 5 | St George | 0.161 | 4 | 124,323 |
| 6 | Keele | 0.144 | 8 | 541,003 |
| 7 | Wilson | 0.143 | 13 | 399,276 |
| 8 | Dufferin | 0.134 | 6 | 258,428 |
| 9 | Bathurst | 0.132 | 6 | 215,520 |
| 10 | Mount Dennis | 0.113 | 13 | 750,881 |
| 11 | Finch | 0.113 | 14 | 741,290 |
| 12 | Sheppard West | 0.093 | 10 | 331,148 |
| 13 | Kipling | 0.091 | 14 | 268,166 |
| 14 | Glencairn | 0.075 | 2 | 45,002 |
| 15 | Scarborough Centre | 0.066 | 16 | 474,233 |
| 16 | Leslie | 0.065 | 5 | 100,053 |
| 17 | McCowan | 0.026 | 10 | 331,097 |
| 18 | Pioneer Village | 0.024 | 12 | 782,786 |
| 19 | Union | 0.022 | 9 | 279,634 |

### 4.3 Hubs vs street stops (plot: `C1_hub_vs_street_betweenness.png`)

- **19 major hubs** (navy) vs **~7,000 street stops** (blue).
- Hubs dominate top-20 structural betweenness.
- **High-ranking street stops:**
  - Wilson Ave at Bathurst St West Side (0.113)
  - Wilson Ave at Avenue Rd (0.102)
  - Triton Rd at Borough Dr (0.095)
  - York Mills Rd / Midland Ave corridor stops (0.06–0.07)

### 4.4 York Mills #1 caveat

York Mills cluster = **19 members** including absorbed Yonge/Wilson/York Mills Rd bus stops. Collapsed graph: 2 station neighbors (Lawrence, Sheppard-Yonge), **15** street-stop neighbors.

**Do not** present as “most important station in real life” — topology + clustering artifact plus genuine bus–subway gateway.

### 4.5 Bottleneck language

High-betweenness hubs are **structural bridges** in the hop-count model. Removing them forces long detours. This is **not** the same as highest delay minutes.

---

## 5. Part 4 — Routing efficiency & accessibility

**Metrics:** Closeness, average path length, route fan-out; delay for operational context.

### 5.1 Closeness — all 140 stations

| Rank | Station | Closeness | Betweenness |
|------|---------|-----------|-------------|
| 1 | Eglinton | 0.055 | 0.211 |
| 2 | Lawrence | 0.054 | 0.227 |
| 3 | York Mills | 0.054 | 0.262 |
| 4 | Lawrence West | 0.053 | 0.139 |
| 5 | Avenue | 0.053 | 0.089 |
| 6 | Cedarvale | 0.053 | 0.175 |
| 7 | Mount Pleasant | 0.053 | — |
| 8 | Forest Hill | 0.053 | 0.083 |
| 9 | Glencairn | 0.053 | 0.075 |

**Pattern:** North Yonge / midtown corridor scores high on closeness and betweenness.

### 5.2 Closeness among 19 major hubs

| Rank | Hub | Closeness | Betweenness rank |
|------|-----|-----------|------------------|
| 1 | Eglinton | 0.055 | 1 |
| 2 | Glencairn | 0.053 | 14 |
| 3 | Bloor | 0.052 | 2 |
| 4 | Wilson | 0.052 | 7 |
| 5 | St George | 0.052 | 5 |
| 15 | Kennedy | 0.044 | 4 |
| 19 | McCowan | 0.040 | 17 |

Major-hub closeness: narrow band **~0.04–0.055**. Glencairn = accessible, not a bottleneck.

### 5.3 How structure affects routing efficiency

| Feature | Effect |
|---------|--------|
| Sparsity (0.00038) | Few alternate paths |
| Hub-and-spoke degree | Travel funnels through hubs |
| ~30 hop avg path | Long topological distances |
| Clustering + cliques | Free transfers within stations (1 hop) |
| No transfer penalty | Bus↔subway cheaper than reality |
| Undirected | Direction ignored; connectivity overstated |

### 5.4 Route connectivity (plot: `H8_hub_route_count.png`)

| Hub | Routes | Graph degree |
|-----|--------|--------------|
| **Kennedy** | **25** | 13 |
| Scarborough Centre | 16 | 8 |
| Finch, Kipling | 14 | 14–18 |
| Wilson | 13 | 13 |
| Bloor | 4 | 9 |
| St George | 4 | 5 |
| Glencairn | 2 | 6 |

Kipling has highest graph degree (18) among hubs; mid-pack betweenness.

---

## 6. Structural vs operational — main narrative

### 6.1 Definitions

| | Structural | Operational |
|--|------------|-------------|
| **Metrics** | Betweenness, closeness, density, path length | Total delay, route count, mode split |
| **Source** | GTFS topology | Disruption CSVs |
| **Question** | Where are paths forced through? | Where does service break down? |

### 6.2 Delay ranking (plot: `H4_hub_total_delay.png`)

| Rank | Hub | Total delay (min) | Betweenness rank |
|------|-----|-------------------|------------------|
| 1 | Kennedy | 938,644 | 4 |
| 2 | Pioneer Village | 782,786 | 18 |
| 3 | Mount Dennis | 750,881 | 10 |
| 4 | Finch | 741,290 | 11 |
| 5 | Warden | 544,049 | 3 |
| 6 | Keele | 541,003 | 6 |
| 7 | Scarborough Centre | 474,233 | 15 |
| 8 | Wilson | 399,276 | 7 |
| 9 | Eglinton | 382,002 | 1 |
| 12 | Union | 279,634 | **19** |

### 6.3 Contrasts to use in slides

**High structure, lower relative delay**
- **Bloor** — #2 betweenness, 114k delay, 4 routes
- **St George** — #5 betweenness, 124k delay, 4 routes
- **Eglinton** — #1 betweenness, 382k delay (moderate operationally)

**High delay, lower structure**
- **Pioneer Village** — #2 delay, #18 betweenness; ~all bus
- **Mount Dennis** — #3 delay, #10 betweenness
- **Finch** — #4 delay, #11 betweenness
- **Scarborough Centre** — 16 routes, 474k delay, #15 betweenness

**High on both**
- **Kennedy** — #4 betweenness, #1 delay, 25 routes
- **Warden** — #3 betweenness, #5 delay

**Inverted**
- **Union** — #19 betweenness, 280k delay, highest crowding (86.1)

**Closeness vs betweenness**
- **Glencairn** — #2 closeness, #14 betweenness, lowest delay (45k)
- **Kennedy** — top betweenness, low closeness (~0.044)

### 6.4 Mode split in delays

| Hub | Bus | Streetcar | Subway |
|-----|-----|-----------|--------|
| Pioneer Village | 782,680 | 106 | 0 |
| Mount Dennis | 750,612 | 269 | 0 |
| Kennedy | 929,198 | 552 | 8,894 |
| Finch | 734,508 | 203 | 6,579 |
| Bloor | 102,970 | 118 | 10,884 |
| Bathurst | 164,287 | 49,164 | 2,069 |
| Union | 242,137 | 33,805 | 3,692 |

### 6.5 Crowding (2024 pax/vehicle-hour)

Top: Union (86.1), Warden/Scarborough Centre/McCowan (84.7), Wilson (84.2), Finch (81.0). Bloor: no match.

---

## 7. Topology insights

### 7.1 Downtown under-ranks on betweenness

Union, Bloor-Yonge, St George sit in a **dense mesh**. Path load is shared across many stations.

### 7.2 Eastern termini

Kennedy, Warden, Victoria Park bridge Scarborough ↔ rest of network → high betweenness and operational exposure.

### 7.3 Bloor-Yonge merge

One physical interchange; GTFS split Line 1 (`Bloor`) vs Line 2 (`Yonge` at Bloor). Merged = 6 stops.

### 7.4 St George

Line 2 sample trip: St George → Yonge (not through old Bloor cluster). St George = Line 1↔2 junction in model.

---

## 8. Limitations

1. **464-trip skeleton** — topology template, not full frequency.
2. **Undirected, unweighted** — no waits, direction, or schedule.
3. **No transfer penalty** — bus↔subway = 1 hop if edge exists.
4. **150 m clustering** — bus stops absorbed into supernodes (York Mills, Eglinton).
5. **k=500** betweenness at stop level (exact at station level).
6. **Historical disruptions** — bus-heavy skew.
7. **Type II ≠ ridership** — Union is the clearest example.

---

## 9. Four presentation parts — what to report

| Part | Course focus | Plot | Top finding |
|------|--------------|------|-------------|
| **1** | Network property analysis | **N3** | Hub-and-spoke; degree-2 peak; sparse + clustered |
| **2** | Station hub identification | (text) | 140 clusters; 19 hubs; cliques matter |
| **3** | Connectivity & bottlenecks | **C1** | Eglinton/Bloor/Warden/Kennedy/St George; hubs beat streets |
| **4** | Structure → routing & accessibility | **H4** or **H8** | Closeness spine; Kennedy 25 routes; Union inversion |

---

## 10. Slide-ready bullets

**Part 1:** 8,056 stops, 12,472 edges, density 0.00038, clustering 0.098, ~30 hops. N3: ~74% at degree 2, hub tail.

**Part 2:** 140 collapsed clusters; 19 major hubs; 3,360 clique edges connect platforms.

**Part 3:** Eglinton, Bloor-Yonge, Warden, Kennedy, St George top bottlenecks among majors. C1: hubs dominate; Wilson corridor competes.

**Part 4:** Eglinton/Lawrence/York Mills lead closeness. Kennedy 25 routes. Structure ≠ operations: Union last on betweenness; Finch/Pioneer Village top delay.

**Closing:** Type II = where the graph is bridge-like. Operations = where service fails. Kennedy is both; Union and Finch show the split.

---

## 11. Output files

| File | Use for |
|------|---------|
| `type_ii_network_summary.csv` | N1/N2 metrics |
| `type_ii_stops_metrics_full.csv` | Per-stop degree, cluster |
| `type_ii_all_stations_collapsed.csv` | All 140 station rankings |
| `type_ii_major_hubs_collapsed.csv` | 19-hub structural metrics |
| `type_ii_hub_bottleneck_combined.csv` | Structure + delay |
| `type_ii_ranking_pool.csv` | P1, C1 comparisons |
| `plots/final/` | N3, C1, H1, H4, H8, P1 |

**Plots kept for report:** N3, N1, N2, C1 (betweenness), H1, H4, H8, P1.

---

## 12. Do not reuse (outdated from earlier runs)

| Old claim | June 6 reality |
|-----------|----------------|
| Sheppard West / Wilson #1 betweenness (among majors) | Eglinton, Bloor, Warden |
| PageRank in Type II | Type I only |
| Separate Bloor and downtown Yonge | Bloor-Yonge Station merged |
| Yonge #1 overall (pre-fix) | York Mills #1 all-station (clustering caveat) |
| `TYPE_II_RESULTS_SUMMARY.md` hub table | Stale — use this file or CSVs |

---

## 13. Scripts

```bash
python typeii.py
python type_ii_final_bars.py
python build_type_ii_slides.py   # optional 2-slide deck
```
