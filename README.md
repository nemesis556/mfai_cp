# AI-Driven Network Stability and Vulnerability Analysis

This repository contains a runnable Python project that implements the architecture described in your proposal:

1. Graph construction from edge list input
2. Adjacency matrix construction
3. Spectral analysis (power iteration for spectral radius)
4. Eigenvector-centrality-like node importance
5. Attack simulation (random, targeted, multi-node)
6. Bayesian collapse-risk estimation
7. First-order style and propositional logic inference
8. GNN-inspired vulnerability scoring model (feature-based classifier)
9. Improvement recommendations
10. Final markdown report generation

## Quick start

```bash
python main.py --input data/sample_edges.txt --seed 7
```

## Input format

One undirected edge per line:

```text
1 2
1 3
2 4
```

## Run tests

```bash
python -m unittest discover -s tests -v
```
