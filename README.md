# TTC Network Analysis

This repository contains the code and data behind the final report *"Understanding the TTC Through Graph Theory: Connectivity, Disruptions, and Network Analysis"*.

It models the TTC as a directed weighted graph built from GTFS stop sequences, then studies routing, centrality, disruption impact, and scalability.

## What Is Included

- `evaluation/type_I/` - graph building, centrality metrics, PageRank, routing comparisons, edge-weight experiments, and disruption simulations
- `evaluation/type_II/` - GTFS stop-time analysis and final bottleneck/disruption visuals
- `evaluation/type_III/` - scalability and graph-structure efficiency experiments
- `website_maps/` - interactive TTC map and metric layers (Please note that the website is not hosted through this repository so some code may be outdated)
- `dataset/` - GTFS feed plus disruption and traffic CSV files
- `outputs/` - generated charts, tables, and route visualizations

## Main Dependencies

- `pandas`
- `networkx`
- `numpy`
- `matplotlib`

## Common Entry Points

- `python3 evaluation/type_I/PageRank/pagerank.py`
- `python3 evaluation/type_I/algoComparison.py`
- `python3 evaluation/type_I/simulations.py`
- `python3 evaluation/type_III/8.3code.py`
- `python3 evaluation/type_III/8.4code.py`

## Notes

- Most scripts read from `dataset/completegtfs/` or `dataset/Complete GTFS/`.
- Outputs are written to `outputs/` or the script-specific folders under `evaluation/`.
- Open `website_maps/ttc_interactive.html` in a browser to explore the interactive map.
