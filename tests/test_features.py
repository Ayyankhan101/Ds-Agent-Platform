import pandas as pd
import json
from pathlib import Path
from ds_platform.agents.features import FeatureEngineer


class TestFeatureEngineer:
    def setup_method(self):
        self.output_dir = Path("data/outputs")
        self.agent = FeatureEngineer(output_dir=str(self.output_dir))
        self.df = pd.DataFrame(
            {
                "date_col": ["2024-01-01", "2024-02-01", "2024-03-01"],
                "num_a": [10, 20, 30],
                "num_b": [5, 10, 15],
                "category": ["x", "y", "z"],
            }
        )

    def test_date_expansion(self):
        config = {"date_columns": ["date_col"], "ratio_pairs": [], "bins": []}
        result = self.agent.transform(self.df, config)
        assert "date_col_year" in result.columns
        assert "date_col_month" in result.columns

    def test_ratio_creation(self):
        config = {"date_columns": [], "ratio_pairs": [("num_a", "num_b")], "bins": []}
        result = self.agent.transform(self.df, config)
        assert "num_a_num_b_ratio" in result.columns
        assert abs(result["num_a_num_b_ratio"].iloc[0] - 2.0) < 0.01

    def test_binning(self):
        df_with_score = pd.DataFrame({"score": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]})
        config = {"date_columns": [], "ratio_pairs": [], "bins": [("score", [0, 50, 100])]}
        result = self.agent.transform(df_with_score, config)
        assert "score_binned" in result.columns

    def test_metadata_saved(self):
        config = {"date_columns": [], "ratio_pairs": [("num_a", "num_b")], "bins": []}
        self.agent.transform(self.df, config)
        metadata_path = self.output_dir / "feature_metadata.json"
        assert metadata_path.exists()
        with open(metadata_path) as f:
            metadata = json.load(f)
        assert len(metadata) > 0
        assert metadata[0]["type"] == "ratio"

    def test_invalid_columns_ignored(self):
        config = {"date_columns": ["nonexistent"], "ratio_pairs": [("x", "y")], "bins": []}
        result = self.agent.transform(self.df, config)
        assert len(result) == len(self.df)
