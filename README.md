# TTC Network Analysis

This repository contains the code and data for the EECS4414 final report, *Understanding the TTC Through Graph Theory: Connectivity, Disruptions, and Network Analysis*.

The project models the TTC as a directed weighted graph built from GTFS stop sequences, then studies routing, centrality, disruption impact, and scalability.

## Project Layout

- `evaluation/type_I/` - graph building, centrality metrics, PageRank, routing comparisons, edge-weight experiments, and disruption simulations
- `evaluation/type_II/scripts_and_results/` - station-cluster analysis, centrality and delay tables, and the final Type II charts
- `evaluation/type_III/` - scalability and graph-structure efficiency experiments
- `website_maps/` - interactive TTC map and generated metric layers
- `dataset/` - GTFS feed plus disruption and traffic CSV files
- `outputs/` - generated charts, tables, and route visualizations

## Quick Start

Install the core Python packages:

```bash
python3 -m pip install pandas networkx numpy matplotlib
```

Then run the scripts you need from the repo root.

## Main Scripts

- `python3 evaluation/type_I/type1_abrar.py` - centrality metrics and static network visuals
- `python3 evaluation/type_I/PageRank/pagerank.py` - PageRank analysis for stops and major hubs
- `python3 evaluation/type_I/diffEdgeWeights.py` - compare trip-frequency, travel-time, and hop-count weights
- `python3 evaluation/type_I/algoComparison.py` - Dijkstra vs A* routing comparison
- `python3 evaluation/type_I/simulations.py` - routing disruption simulations
- `python3 evaluation/type_II/scripts_and_results/typeii.py` - Type II graph, station-cluster, and delay analysis
- `python3 evaluation/type_II/scripts_and_results/type_ii_final_bars.py` - final Type II charts from the generated CSVs
- `python3 evaluation/type_III/8.3code.py` - scalability study
- `python3 evaluation/type_III/8.4code.py` - structure-efficiency study

## Outputs

- Type I figures and tables are written to `outputs/`
- Type II CSVs and summary tables are written to `evaluation/type_II/scripts_and_results/`
- Type II charts are written to `evaluation/type_II/scripts_and_results/plots/`
- Type III outputs are written to `outputs/8.3/` and `outputs/8.4/`

## Notes

- Most scripts read from `dataset/completegtfs/`; several also fall back to `dataset/Complete GTFS/`.
