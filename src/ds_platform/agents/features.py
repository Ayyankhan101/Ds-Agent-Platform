import pandas as pd
import json
from typing import List, Dict, Any
from pathlib import Path

class FeatureEngineer:
    """
    Agent for Feature Engineering: transformations, ratios, and binning.
    """
    def __init__(self, output_dir: str = "data/outputs"):
        self.output_dir = Path(output_dir)
        self.metadata = []

    def transform(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        df_feat = df.copy()

        # 1. Date Features
        date_cols = config.get("date_columns", [])
        for col in date_cols:
            if col in df_feat.columns:
                df_feat[col] = pd.to_datetime(df_feat[col])
                df_feat[f"{col}_year"] = df_feat[col].dt.year
                df_feat[f"{col}_month"] = df_feat[col].dt.month
                self.metadata.append({"feature": col, "type": "date_expansion"})

        # 2. Ratios
        ratios = config.get("ratio_pairs", [])
        for num, den in ratios:
            if num in df_feat.columns and den in df_feat.columns:
                feat_name = f"{num}_{den}_ratio"
                df_feat[feat_name] = df_feat[num] / (df_feat[den] + 1e-9)
                self.metadata.append({"feature": feat_name, "type": "ratio", "components": [num, den]})

        # 3. Binning
        bins_config = config.get("bins", [])
        for col, bin_edges in bins_config:
            if col in df_feat.columns:
                df_feat[f"{col}_binned"] = pd.cut(df_feat[col], bins=bin_edges)
                self.metadata.append({"feature": f"{col}_binned", "type": "binning"})

        # Save metadata
        with open(self.output_dir / "feature_metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=4)

        return df_feat
