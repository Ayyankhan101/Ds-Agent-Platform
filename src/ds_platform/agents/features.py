import pandas as pd
import json
import pickle
from typing import Dict, Any, Optional
from pathlib import Path
import numpy as np
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    LabelEncoder,
    OneHotEncoder,
    PolynomialFeatures,
)


class FeatureEngineer:
    """
    Agent for Feature Engineering: transformations, ratios, binning, scaling, and encoding.
    """

    def __init__(self, output_dir: str = "data/outputs"):
        self.output_dir = Path(output_dir)
        self.metadata = []
        self.encoders: Dict[str, Any] = {}
        self.scaler: Optional[Any] = None

    def transform(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        df_feat = df.copy()

        # 1. Date Features
        date_cols = config.get("date_columns", [])
        for col in date_cols:
            if col in df_feat.columns:
                df_feat[col] = pd.to_datetime(df_feat[col])
                df_feat[f"{col}_year"] = df_feat[col].dt.year
                df_feat[f"{col}_month"] = df_feat[col].dt.month
                df_feat[f"{col}_day"] = df_feat[col].dt.day
                self.metadata.append({"feature": col, "type": "date_expansion"})

        # 2. Ratios
        ratios = config.get("ratio_pairs", [])
        for num, den in ratios:
            if num in df_feat.columns and den in df_feat.columns:
                feat_name = f"{num}_{den}_ratio"
                df_feat[feat_name] = df_feat[num] / (df_feat[den] + 1e-9)
                self.metadata.append(
                    {"feature": feat_name, "type": "ratio", "components": [num, den]}
                )

        # 3. Binning
        bins_config = config.get("bins", [])
        for col, bin_edges in bins_config:
            if col in df_feat.columns:
                df_feat[f"{col}_binned"] = pd.cut(df_feat[col], bins=bin_edges)
                self.metadata.append({"feature": f"{col}_binned", "type": "binning"})

        # 4. Scaling
        scale_method = config.get("scaling", "None")
        if scale_method != "None":
            df_feat = self.apply_scaling(df_feat, scale_method)

        # 5. Encoding
        encode_method = config.get("encoding", "None")
        target_col = config.get("target")
        if encode_method != "None" and target_col:
            df_feat = self.apply_encoding(df_feat, encode_method, target_col)

        # Save metadata
        with open(self.output_dir / "feature_metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=4)

        return df_feat

    def apply_scaling(self, df: pd.DataFrame, method: str) -> pd.DataFrame:
        """Apply scaling to numeric columns."""
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

        if not numeric_cols:
            return df

        if method == "StandardScaler":
            scaler = StandardScaler()
        elif method == "MinMaxScaler":
            scaler = MinMaxScaler()
        elif method == "RobustScaler":
            scaler = RobustScaler()
        else:
            return df

        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        self.scaler = scaler

        # Save scaler for inference
        with open(self.output_dir / "scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

        self.metadata.append({"feature": "all_numeric", "type": f"scaling_{method.lower()}"})
        return df

    def apply_encoding(self, df: pd.DataFrame, method: str, target_col: str) -> pd.DataFrame:
        """Apply encoding to categorical columns."""
        cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

        # Remove target column from encoding if it's categorical
        cat_cols = [c for c in cat_cols if c != target_col]

        if not cat_cols:
            return df

        if method == "Label":
            for col in cat_cols:
                le = LabelEncoder()
                df[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))
                self.encoders[col] = le
                self.metadata.append({"feature": col, "type": "label_encoding"})

        elif method == "OneHot":
            df = pd.get_dummies(df, columns=cat_cols, prefix=cat_cols)
            for col in cat_cols:
                self.metadata.append({"feature": col, "type": "onehot_encoding"})

        elif method == "Target" and target_col in df.columns:
            # Target encoding - only works well with binary target
            if df[target_col].nunique() <= 2:
                target_means = df.groupby(target_col).mean(numeric_only=True)
                for col in cat_cols:
                    mapping = df.groupby(col)[target_col].mean()
                    df[f"{col}_target_enc"] = df[col].map(mapping)
                    self.metadata.append({"feature": col, "type": "target_encoding"})
            else:
                # Fallback to label encoding if target has more than 2 values
                for col in cat_cols:
                    le = LabelEncoder()
                    df[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))
                    self.encoders[col] = le

        # Save encoders
        with open(self.output_dir / "encoders.pkl", "wb") as f:
            pickle.dump(self.encoders, f)

        return df

    def create_interaction_features(
        self,
        df: pd.DataFrame,
        columns: list,
        include_products: bool = True,
        include_powers: bool = False,
        include_exponentials: bool = False,
    ) -> pd.DataFrame:
        """
        Create interaction and polynomial features.

        Args:
            df: DataFrame
            columns: Columns to create interactions from
            include_products: Create col1 * col2 interactions
            include_powers: Create squared terms (col^2)
            include_exponentials: Create exponential terms

        Returns:
            DataFrame with new features
        """
        df_new = df.copy()

        for i, col1 in enumerate(columns):
            if col1 not in df_new.columns:
                continue

            # Squared terms
            if include_powers:
                df_new[f"{col1}_squared"] = df_new[col1] ** 2
                self.metadata.append(
                    {"feature": f"{col1}_squared", "type": "polynomial", "degree": 2}
                )

            # Exponential
            if include_exponentials:
                df_new[f"{col1}_exp"] = np.exp(df_new[col1])
                self.metadata.append({"feature": f"{col1}_exp", "type": "exponential"})

            # Interaction with other columns
            if include_products:
                for col2 in columns[i + 1 :]:
                    if col2 not in df_new.columns:
                        continue

                    # Product interaction
                    feat_name = f"{col1}_x_{col2}"
                    df_new[feat_name] = df_new[col1] * df_new[col2]
                    self.metadata.append(
                        {"feature": feat_name, "type": "interaction", "components": [col1, col2]}
                    )

        return df_new

    def create_log_features(
        self,
        df: pd.DataFrame,
        columns: list,
    ) -> pd.DataFrame:
        """
        Create log-transformed features for skewed data.

        Args:
            df: DataFrame
            columns: Columns to log-transform

        Returns:
            DataFrame with log features
        """
        df_log = df.copy()

        for col in columns:
            if col not in df_log.columns:
                continue

            # Log1p for handling zeros
            df_log[f"{col}_log"] = np.log1p(df_log[col])
            self.metadata.append({"feature": f"{col}_log", "type": "log_transform"})

        return df_log

    def create_aggregated_features(
        self,
        df: pd.DataFrame,
        numeric_col: str,
        categorical_col: str,
        operations: list = None,
    ) -> pd.DataFrame:
        """
        Create aggregated features from groupby operations.

        Args:
            df: DataFrame
            numeric_col: Numeric column to aggregate
            categorical_col: Column to group by
            operations: List of operations ('mean', 'sum', 'count', 'std')

        Returns:
            DataFrame with aggregated features
        """
        if operations is None:
            operations = ["mean", "sum", "std"]

        df_agg = df.copy()

        for op in operations:
            new_col = f"{numeric_col}_by_{categorical_col}_{op}"
            df_agg[new_col] = df.groupby(categorical_col)[numeric_col].transform(op)
            self.metadata.append(
                {
                    "feature": new_col,
                    "type": f"groupby_{op}",
                    "components": [numeric_col, categorical_col],
                }
            )

        return df_agg
