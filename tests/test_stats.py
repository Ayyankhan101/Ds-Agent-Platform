import pandas as pd
from ds_platform.agents.stats import StatsAgent


class TestStatsAgent:
    def setup_method(self):
        self.agent = StatsAgent()
        self.df = pd.DataFrame(
            {
                "value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "category": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
                "result": ["yes", "no", "yes", "no", "yes", "no", "yes", "no", "yes", "no"],
            }
        )

    def test_ttest_returns_statistic_and_pvalue(self):
        result = self.agent.run_test(self.df, "t-test", {"column": "value", "popmean": 5})
        assert "statistic" in result
        assert "p_value" in result
        assert "interpretation" in result

    def test_ttest_interpretation(self):
        result = self.agent.run_test(self.df, "t-test", {"column": "value", "popmean": 5})
        assert result["interpretation"] in ["Reject H0", "Fail to reject H0"]

    def test_chi_square_returns_stats(self):
        result = self.agent.run_test(
            self.df, "chi-square", {"column1": "category", "column2": "result"}
        )
        assert "statistic" in result
        assert "p_value" in result
        assert "dof" in result

    def test_chi_square_interpretation(self):
        result = self.agent.run_test(
            self.df, "chi-square", {"column1": "category", "column2": "result"}
        )
        assert result["interpretation"] in ["Reject H0", "Fail to reject H0"]

    def test_invalid_column_handled(self):
        result = self.agent.run_test(self.df, "t-test", {"column": "nonexistent"})
        assert "error" in result

    def test_invalid_test_type_handled(self):
        result = self.agent.run_test(self.df, "invalid_test", {"column": "value"})
        assert "error" in result
