import pandas as pd
import numpy as np
import json
from typing import Dict, Any, Optional
from pathlib import Path

class CleaningPipeline:
    """
    Agent for dataset cleaning: imputation, outlier removal, and encoding.
    """
    def __init__(self, output_dir: str = "data/outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log = []

    def clean(self, df: pd.DataFrame, strategy: Dict[str, str], outlier_config: Dict[str, Any]) -> pd.DataFrame:
        """
        Processes the dataframe based on user-defined strategies.
        """
        df_cleaned = df.copy()
        
        # 1. Imputation
        for col, method in strategy.items():
            if col in df_cleaned.columns:
                if method == "mean":
                    df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mean())
                elif method == "median":
                    df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
                elif method == "mode":
                    df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])
                self.log.append({"task": "imputation", "column": col, "method": method})

        # 2. Outlier Removal (Simple IQR implementation)
        if outlier_config.get("method") == "IQR":
            threshold = outlier_config.get("threshold", 1.5)
            # Only numeric columns
            numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                Q1 = df_cleaned[col].quantile(0.25)
                Q3 = df_cleaned[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
                self.log.append({"task": "outlier_removal", "column": col, "method": "IQR", "threshold": threshold})

        # Save outputs
        df_cleaned.to_csv(self.output_dir / "cleaned_data.csv", index=False)
        with open(self.output_dir / "transformation_log.json", "w") as f:
            json.dump(self.log, f, indent=4)
            
        return df_cleaned
