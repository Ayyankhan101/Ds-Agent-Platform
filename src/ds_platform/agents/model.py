import pandas as pd
import numpy as np
import json
import pickle
from typing import Dict, Any, Optional
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    confusion_matrix,
    classification_report,
)


class ModelTrainer:
    """
    Agent for Machine Learning: Training, evaluation, and result tracking.
    """

    def __init__(self, output_dir: str = "data/outputs"):
        self.output_dir = Path(output_dir)
        self.model: Optional[Any] = None

    def train_and_evaluate(
        self, df: pd.DataFrame, target: str, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        X = df.drop(columns=[target])
        y = df[target]

        # Handle non-numeric columns
        X = X.select_dtypes(include=["number"])

        test_size = config.get("test_size", 0.2)
        random_state = config.get("random_seed", 42)
        problem_type = config.get("problem_type", "Classification")
        model_choice = config.get("model_type", "Random Forest")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Select model based on choice and problem type
        self.model = self._get_model(model_choice, problem_type, random_state)

        # Train model
        self.model.fit(X_train, y_train)

        # Make predictions
        y_pred = self.model.predict(X_test)

        # Calculate metrics
        if problem_type == "Classification":
            metrics = self._classification_metrics(y_test, y_pred)
        else:
            metrics = self._regression_metrics(y_test, y_pred)

        # Get feature importances (if available)
        feature_importances = None
        if hasattr(self.model, "feature_importances_"):
            feature_importances = dict(zip(X.columns, self.model.feature_importances_.tolist()))

        # Save model
        with open(self.output_dir / "model.pkl", "wb") as f:
            pickle.dump(self.model, f)

        results = {
            "config": config,
            "metrics": metrics,
            "feature_importances": feature_importances,
            "model": self.model,
            "y_test": y_test,
            "y_pred": y_pred,
        }

        with open(self.output_dir / "model_results.json", "w") as f:
            # Don't pickle model for JSON
            json_safe = {k: v for k, v in results.items() if k != "model"}
            json.dump(json_safe, f, indent=4, default=str)

        return results

    def _get_model(self, model_choice: str, problem_type: str, random_state: int):
        """Get model instance based on choice."""
        if problem_type == "Classification":
            model_map = {
                "Random Forest": RandomForestClassifier(random_state=random_state),
                "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
                "Logistic Regression": LogisticRegression(random_state=random_state, max_iter=1000),
                "SVM": SVC(random_state=random_state),
            }
        else:  # Regression
            model_map = {
                "Random Forest": RandomForestRegressor(random_state=random_state),
                "Gradient Boosting": GradientBoostingRegressor(random_state=random_state),
                "Ridge": Ridge(random_state=random_state),
                "Lasso": Lasso(random_state=random_state),
            }

        return model_map.get(model_choice, RandomForestClassifier(random_state=random_state))

    def _classification_metrics(self, y_test, y_pred) -> Dict[str, Any]:
        """Calculate classification metrics."""
        cm = confusion_matrix(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)

        return {
            "accuracy": float(accuracy),
            "confusion_matrix": cm.tolist(),
        }

    def _regression_metrics(self, y_test, y_pred) -> Dict[str, Any]:
        """Calculate regression metrics."""
        mse = mean_squared_error(y_test, y_pred)
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(y_test - y_pred)))

        return {
            "rmse": rmse,
            "mae": mae,
            "mse": float(mse),
        }
