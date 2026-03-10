import unittest

from main import adjacency_matrix, analyze, build_graph, load_edges, power_iteration


class PipelineTests(unittest.TestCase):
    def test_load_and_graph(self):
        edges = load_edges("data/sample_edges.txt")
        g = build_graph(edges)
        self.assertIn(1, g)
        self.assertIn(2, g[1])

    def test_spectral_radius_positive(self):
        g = build_graph(load_edges("data/sample_edges.txt"))
        _, a = adjacency_matrix(g)
        radius, _ = power_iteration(a)
        self.assertGreater(radius, 0.0)

    def test_report_contains_sections(self):
        report = analyze(load_edges("data/sample_edges.txt"), seed=7)
        self.assertIn("# Network Stability Report", report)
        self.assertIn("## Bayesian Collapse Probability", report)
        self.assertIn("## Recommendations", report)


if __name__ == "__main__":
    unittest.main()
