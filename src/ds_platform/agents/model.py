import pandas as pd
import json
from typing import Dict, Any
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix

class ModelTrainer:
    """
    Agent for Machine Learning: Training, evaluation, and result tracking.
    """
    def __init__(self, output_dir: str = "data/outputs"):
        self.output_dir = Path(output_dir)

    def train_and_evaluate(self, df: pd.DataFrame, target: str, config: Dict[str, Any]) -> Dict[str, Any]:
        X = df.drop(columns=[target])
        y = df[target]
        
        test_size = config.get("test_size", 0.2)
        random_state = config.get("random_seed", 42)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Simple Logic to choose model
        if y.dtype == 'object' or y.nunique() < 10:
            model = RandomForestClassifier(random_state=random_state)
            model.fit(X_train.select_dtypes(include=['number']), y_train)
            preds = model.predict(X_test.select_dtypes(include=['number']))
            metrics = {"accuracy": accuracy_score(y_test, preds)}
        else:
            model = RandomForestRegressor(random_state=random_state)
            model.fit(X_train.select_dtypes(include=['number']), y_train)
            preds = model.predict(X_test.select_dtypes(include=['number']))
            metrics = {"rmse": float(np.sqrt(mean_squared_error(y_test, preds)))}

        results = {
            "config": config,
            "metrics": metrics
        }
        
        with open(self.output_dir / "model_results.json", "w") as f:
            json.dump(results, f, indent=4)
            
        return results
import numpy as np # Needed for sqrt
