# ML for Layer Assignment using Hybrid GCN–CNN Model

This repository provides a framework for ML-based layer assignment in Global Routing. It includes code for generating data using traditional layer assignment algorithms, as well as scripts for training and evaluating a **Hybrid GCN–CNN architecture** designed specifically for layer assignment.  
Both traditional and ML-based layer assignment guides are evaluated to measure their impact on global routing and detailed routing metrics.

---

## Results

The `results/` directory contains `.xlsx` files that report routing performance after running the layer assignment guides through global and detailed routing using **[OpenROAD](https://theopenroadproject.org/)**.  
Results are provided for both ML-based and traditional timing-driven and congestion-driven layer assignment algorithms across all datapoints in the **ASAP7** and **Nangate45** technology nodes.

The file **`datapoints_details.xlsx`** summarizes the attributes of each datapoint and documents how these values were generated using **[OpenROAD-Flow-Scripts](https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts)**.

---






