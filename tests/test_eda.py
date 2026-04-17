import pandas as pd
import matplotlib.pyplot as plt
from ds_platform.agents.eda import EDAAgent


class TestEDAAgent:
    def setup_method(self):
        self.agent = EDAAgent(output_dir="data/reports")
        self.df = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5],
                "b": [5, 4, 3, 2, 1],
                "c": ["x", "y", "z", "x", "y"],
                "target": [10, 20, 30, 40, 50],
            }
        )

    def test_analyze_returns_correlation(self):
        results = self.agent.analyze(self.df, target="target", correlation_method="pearson")
        assert "correlation" in results
        assert "a" in results["correlation"]
        assert "b" in results["correlation"]

    def test_analyze_returns_summary(self):
        results = self.agent.analyze(self.df, target="target", correlation_method="pearson")
        assert "summary" in results
        assert "a" in results["summary"]
        assert results["summary"]["a"]["mean"] == 3.0

    def test_analyze_returns_target_info(self):
        results = self.agent.analyze(self.df, target="target", correlation_method="pearson")
        assert "target_info" in results
        assert "skew" in results["target_info"]
        assert results["target_info"]["unique_values"] == 5

    def test_analyze_spearman_correlation(self):
        results = self.agent.analyze(self.df, target="target", correlation_method="spearman")
        assert "correlation" in results

    def test_plot_correlations_returns_figure(self):
        fig = self.agent.plot_correlations(self.df)
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
