import pandas as pd
import json
from pathlib import Path
from ds_platform.agents.model import ModelTrainer


class TestModelTrainer:
    def setup_method(self):
        self.output_dir = Path("data/outputs")
        self.agent = ModelTrainer(output_dir=str(self.output_dir))

    def test_classifier_for_categorical_target(self):
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "feature2": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                "target": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
            }
        )
        result = self.agent.train_and_evaluate(df, "target", {"test_size": 0.2, "random_seed": 42})
        assert "metrics" in result
        assert "accuracy" in result["metrics"]

    def test_regressor_for_numeric_target(self):
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "feature2": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                "target": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            }
        )
        result = self.agent.train_and_evaluate(
            df, "target", {"test_size": 0.2, "random_seed": 42, "problem_type": "Regression"}
        )
        assert "metrics" in result
        assert "rmse" in result["metrics"]

    def test_classifier_for_low_unique_count(self):
        df = pd.DataFrame(
            {"feature1": list(range(100)), "target": [1 if i < 50 else 0 for i in range(100)]}
        )
        result = self.agent.train_and_evaluate(df, "target", {"test_size": 0.2, "random_seed": 42})
        assert "metrics" in result
        assert "accuracy" in result["metrics"]

    def test_config_options(self):
        df = pd.DataFrame({"feature1": list(range(20)), "target": list(range(20))})
        result = self.agent.train_and_evaluate(df, "target", {"test_size": 0.3, "random_seed": 123})
        assert result["config"]["test_size"] == 0.3
        assert result["config"]["random_seed"] == 123

    def test_results_saved_to_file(self):
        df = pd.DataFrame({"feature1": list(range(10)), "target": list(range(10))})
        self.agent.train_and_evaluate(df, "target", {"test_size": 0.2, "random_seed": 42})
        results_path = self.output_dir / "model_results.json"
        assert results_path.exists()
        with open(results_path) as f:
            saved = json.load(f)
        assert "metrics" in saved
