# ML for Layer Assignment using Hybrid GCN–CNN Model

This repository provides a framework for ML-based layer assignment in Global Routing. It includes code for generating data using traditional layer assignment algorithms, as well as scripts for training and evaluating a **Hybrid GCN–CNN architecture** designed specifically for layer assignment. Both traditional and ML-based layer assignment guides are evaluated to measure their impact on global routing and detailed routing metrics. 

![Alt text](twoflows.png)

---
## Abstract
Timing-driven layer assignment during global routing significantly impacts delay and congestion. Traditional methods rely on iterative heuristics with repeated timing and congestion analysis under the hood, leading to high runtime and limited scalability for modern designs. We present a hybrid machine learning (ML) framework that combines graph convolutional networks (GCNs) and convolutional neural networks (CNNs) for fast, timing- and congestion-driven layer assignment. GCNs model netlist connectivity and timing-critical paths, while CNNs extract spatial features such as capacity and utilization maps across the metal layer stack from a placed layout. Formulated as a multi-class classification problem, our model predicts the layer for every Steiner tree edge of a net. The ML model is trained on timing and congestion-driven global-routed layout data to predict optimized layer assignments rapidly in a single pass. Experiments on benchmark designs demonstrate that our approach improves worst-case and total negative slack post detailed route while achieving a speedup compared to traditional congestion and timing-driven methods.
## Results

The `results/` directory contains `.xlsx` files that report routing performance after running the layer assignment guides through global and detailed routing using **[OpenROAD](https://theopenroadproject.org/)**.  
Results are provided for both ML-based and traditional timing-driven and congestion-driven layer assignment algorithms across all datapoints in the **ASAP7** and **Nangate45** technology nodes.

The file **`datapoints_details.xlsx`** summarizes the attributes of each datapoint and documents how these values were generated using **[OpenROAD-Flow-Scripts](https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts)**.

---






