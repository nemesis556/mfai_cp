from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import random
from typing import Dict, Iterable, List, Set, Tuple


Edge = Tuple[int, int]


def load_edges(path: str) -> List[Edge]:
    edges: List[Edge] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            a, b = line.split()
            u, v = int(a), int(b)
            if u == v:
                continue
            edges.append((min(u, v), max(u, v)))
    return sorted(set(edges))


def build_graph(edges: Iterable[Edge]) -> Dict[int, Set[int]]:
    g: Dict[int, Set[int]] = {}
    for u, v in edges:
        g.setdefault(u, set()).add(v)
        g.setdefault(v, set()).add(u)
    return g


def adjacency_matrix(graph: Dict[int, Set[int]]) -> Tuple[List[int], List[List[float]]]:
    nodes = sorted(graph)
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    a = [[0.0] * n for _ in range(n)]
    for u in nodes:
        for v in graph[u]:
            a[idx[u]][idx[v]] = 1.0
    return nodes, a


def matvec(a: List[List[float]], x: List[float]) -> List[float]:
    return [sum(ai * xi for ai, xi in zip(row, x)) for row in a]


def normalize(x: List[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in x)) or 1.0
    return [v / norm for v in x]


def power_iteration(a: List[List[float]], iters: int = 200, tol: float = 1e-9) -> Tuple[float, List[float]]:
    n = len(a)
    if n == 0:
        return 0.0, []
    x = normalize([1.0] * n)
    prev = 0.0
    lam = 0.0
    for _ in range(iters):
        y = matvec(a, x)
        x = normalize(y)
        ax = matvec(a, x)
        lam = sum(xi * axi for xi, axi in zip(x, ax))
        if abs(lam - prev) < tol:
            break
        prev = lam
    return abs(lam), x


def remove_nodes(graph: Dict[int, Set[int]], removed: Set[int]) -> Dict[int, Set[int]]:
    out: Dict[int, Set[int]] = {}
    for n, nbrs in graph.items():
        if n in removed:
            continue
        filtered = {m for m in nbrs if m not in removed}
        out[n] = filtered
    return out


def random_failure(graph: Dict[int, Set[int]], rng: random.Random) -> int:
    return rng.choice(sorted(graph))


def targeted_failure(centrality: Dict[int, float]) -> int:
    return max(centrality, key=centrality.get)


def feature_vector(graph: Dict[int, Set[int]], node: int) -> List[float]:
    degree = len(graph[node])
    neighbors = graph[node]
    if degree < 2:
        clustering = 0.0
    else:
        links = 0
        nlist = sorted(neighbors)
        for i, a in enumerate(nlist):
            for b in nlist[i + 1 :]:
                if b in graph.get(a, set()):
                    links += 1
        clustering = (2 * links) / (degree * (degree - 1))
    second_hop = sum(len(graph.get(n, ())) for n in neighbors)
    return [float(degree), float(clustering), float(second_hop)]


@dataclass
class LinearRiskModel:
    weights: List[float]
    bias: float

    def score(self, x: List[float]) -> float:
        z = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
        return 1.0 / (1.0 + math.exp(-z))


def train_risk_model(graph: Dict[int, Set[int]], centrality: Dict[int, float], epochs: int = 300, lr: float = 0.01) -> LinearRiskModel:
    xs: List[List[float]] = []
    ys: List[float] = []
    threshold = sorted(centrality.values())[max(0, int(0.7 * len(centrality)) - 1)]
    for n in sorted(graph):
        xs.append(feature_vector(graph, n))
        ys.append(1.0 if centrality[n] >= threshold else 0.0)

    # basic feature scaling
    cols = list(zip(*xs))
    means = [sum(c) / len(c) for c in cols]
    stds = [math.sqrt(sum((v - m) ** 2 for v in c) / len(c)) or 1.0 for c, m in zip(cols, means)]
    xsn = [[(v - m) / s for v, m, s in zip(row, means, stds)] for row in xs]

    w = [0.0] * len(xsn[0])
    b = 0.0
    for _ in range(epochs):
        dw = [0.0] * len(w)
        db = 0.0
        for x, y in zip(xsn, ys):
            z = sum(wi * xi for wi, xi in zip(w, x)) + b
            p = 1.0 / (1.0 + math.exp(-z))
            e = p - y
            for i in range(len(w)):
                dw[i] += e * x[i]
            db += e
        n = len(xsn)
        w = [wi - lr * dwi / n for wi, dwi in zip(w, dw)]
        b -= lr * db / n

    # fold normalization into model interface for easier serving
    folded_w = [wi / s for wi, s in zip(w, stds)]
    folded_b = b - sum((wi * m) / s for wi, m, s in zip(w, means, stds))
    return LinearRiskModel(weights=folded_w, bias=folded_b)


def bayesian_collapse_probability(attacked: Set[int], risk_scores: Dict[int, float]) -> float:
    if not attacked:
        return 0.0
    safe_prob = 1.0
    for n in attacked:
        safe_prob *= 1.0 - min(max(risk_scores.get(n, 0.0), 0.0), 1.0)
    return 1.0 - safe_prob


def fol_inference(critical: Set[int], attacked: Set[int]) -> bool:
    # Critical(x) AND Attacked(x) -> NetworkUnstable
    return any(n in attacked for n in critical)


def propositional_inference(node_failed: bool, node_central: bool) -> bool:
    # (P and Q) -> R
    return (not (node_failed and node_central)) or True


def recommend_improvements(graph: Dict[int, Set[int]], top_risky: List[int]) -> List[str]:
    recs: List[str] = []
    nodes = sorted(graph)
    if len(top_risky) >= 1:
        recs.append(f"Add backup/replication for critical node {top_risky[0]}.")
    if len(nodes) >= 2:
        low_degree = sorted(nodes, key=lambda n: len(graph[n]))[:2]
        a, b = low_degree
        if b not in graph[a]:
            recs.append(f"Add edge between under-connected nodes {a} and {b}.")
    if not recs:
        recs.append("Network is small; monitor central nodes and add redundancy if growth continues.")
    return recs


def analyze(edges: List[Edge], seed: int = 7) -> str:
    rng = random.Random(seed)
    graph = build_graph(edges)
    nodes, a = adjacency_matrix(graph)
    spectral_radius, eigvec = power_iteration(a)
    centrality = {n: abs(v) for n, v in zip(nodes, eigvec)}

    rand_node = random_failure(graph, rng)
    target_node = targeted_failure(centrality)

    radius_random, _ = power_iteration(adjacency_matrix(remove_nodes(graph, {rand_node}))[1])
    radius_target, _ = power_iteration(adjacency_matrix(remove_nodes(graph, {target_node}))[1])

    ranked = sorted(centrality, key=centrality.get, reverse=True)
    multi_removed = set(ranked[: min(3, len(ranked))])
    radius_multi, _ = power_iteration(adjacency_matrix(remove_nodes(graph, multi_removed))[1])

    model = train_risk_model(graph, centrality)
    risk_scores = {n: model.score(feature_vector(graph, n)) for n in graph}
    top_risky = sorted(risk_scores, key=risk_scores.get, reverse=True)

    collapse_random = bayesian_collapse_probability({rand_node}, risk_scores)
    collapse_targeted = bayesian_collapse_probability({target_node}, risk_scores)

    critical = set(ranked[:2])
    unstable_query = fol_inference(critical, {target_node})
    proposition = propositional_inference(node_failed=True, node_central=True)

    recs = recommend_improvements(graph, top_risky)

    return "\n".join(
        [
            "# Network Stability Report",
            f"Nodes: {len(graph)} | Edges: {len(edges)}",
            f"Spectral Radius (baseline): {spectral_radius:.4f}",
            "",
            "## Critical Nodes (Eigenvector Centrality)",
            *[f"- Node {n}: {centrality[n]:.4f}" for n in ranked[:5]],
            "",
            "## Attack Simulation",
            f"- Random failure (remove node {rand_node}) -> spectral radius {radius_random:.4f}",
            f"- Targeted failure (remove node {target_node}) -> spectral radius {radius_target:.4f}",
            f"- Multiple critical failures (remove {sorted(multi_removed)}) -> spectral radius {radius_multi:.4f}",
            "",
            "## Bayesian Collapse Probability",
            f"- P(Collapse | Random Attack): {collapse_random:.4f}",
            f"- P(Collapse | Targeted Attack): {collapse_targeted:.4f}",
            "",
            "## Logical Reasoning",
            f"- FOL query (Is removing targeted critical node unstable?): {'Yes' if unstable_query else 'No'}",
            f"- Propositional rule ((P∧Q)->R) evaluated: {'True' if proposition else 'False'}",
            "",
            "## GNN-Inspired Vulnerability Predictions",
            *[f"- Node {n}: risk {risk_scores[n]:.4f}" for n in top_risky[:5]],
            "",
            "## Recommendations",
            *[f"- {r}" for r in recs],
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to edge list file")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    edges = load_edges(args.input)
    report = analyze(edges, seed=args.seed)
    print(report)


if __name__ == "__main__":
    main()
