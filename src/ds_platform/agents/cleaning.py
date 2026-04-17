"""
Advanced Data Cleaning Pipeline.
Imputation strategies, outlier detection (IQR & Z-score), encoding, and transformation tracking.
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from scipy import stats


class CleaningPipeline:
    """
    Agent for dataset cleaning: imputation, outlier removal, and encoding.

    Features:
    - Multiple imputation strategies (mean, median, mode, forward_fill, group-based)
    - IQR and Z-score outlier detection
    - Label and one-hot encoding
    - Comprehensive transformation logging
    """

    def __init__(self, output_dir: str = "data/outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log: List[Dict[str, Any]] = []
        self.null_count_before: int = 0
        self.null_count_after: int = 0
        self.shape_before: Tuple[int, int] = (0, 0)

    def clean(
        self,
        df: pd.DataFrame,
        strategy: Dict[str, str],
        outlier_config: Dict[str, Any],
        group_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Process the dataframe based on user-defined strategies.

        Args:
            df: Input DataFrame
            strategy: Dict mapping column -> imputation method
            outlier_config: Dict with 'method' ('IQR' or 'Zscore'), 'threshold'
            group_cols: Optional list of columns for group-based imputation

        Returns:
            Cleaned DataFrame
        """
        df_cleaned = df.copy()

        # Track initial state
        self.null_count_before = df_cleaned.isnull().sum().sum()
        self.shape_before = df_cleaned.shape

        # 1. Imputation
        df_cleaned = self._apply_imputation(df_cleaned, strategy, group_cols)

        # 2. Outlier Removal
        df_cleaned = self._apply_outlier_removal(df_cleaned, outlier_config)

        # 3. Final state tracking
        self.null_count_after = df_cleaned.isnull().sum().sum()

        # Save outputs
        df_cleaned.to_csv(self.output_dir / "cleaned_data.csv", index=False)
        self._save_transformation_log()

        return df_cleaned

    def _apply_imputation(
        self,
        df: pd.DataFrame,
        strategy: Dict[str, str],
        group_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Apply imputation strategies to DataFrame."""
        df_imputed = df.copy()

        for col, method in strategy.items():
            if col not in df_imputed.columns:
                continue

            null_count = df_imputed[col].isnull().sum()
            if null_count == 0:
                continue

            if method == "mean":
                fill_value = df_imputed[col].mean()
                df_imputed[col] = df_imputed[col].fillna(fill_value)
            elif method == "median":
                fill_value = df_imputed[col].median()
                df_imputed[col] = df_imputed[col].fillna(fill_value)
            elif method == "mode":
                fill_value = df_imputed[col].mode()[0]
                df_imputed[col] = df_imputed[col].fillna(fill_value)
            elif method == "forward_fill":
                df_imputed[col] = df_imputed[col].fillna(method="ffill")
            elif method == "backward_fill":
                df_imputed[col] = df_imputed[col].fillna(method="bfill")
            elif method == "group_mean" and group_cols:
                df_imputed = self._group_imputation(df_imputed, col, group_cols, "mean")
            elif method == "group_median" and group_cols:
                df_imputed = self._group_imputation(df_imputed, col, group_cols, "median")

            self.log.append(
                {
                    "task": "imputation",
                    "column": col,
                    "method": method,
                    "nulls_filled": int(null_count),
                }
            )

        return df_imputed

    def _group_imputation(
        self,
        df: pd.DataFrame,
        target_col: str,
        group_cols: List[str],
        agg_func: str = "mean",
    ) -> pd.DataFrame:
        """
        Impute missing values based on group statistics.

        Args:
            df: DataFrame
            target_col: Column to impute
            group_cols: Columns to group by
            agg_func: 'mean' or 'median'

        Returns:
            DataFrame with imputed values
        """
        df_grouped = df.copy()

        # Calculate group statistics
        if agg_func == "mean":
            group_means = df_grouped.groupby(group_cols)[target_col].transform("mean")
        else:
            group_means = df_grouped.groupby(group_cols)[target_col].transform("median")

        # Fill nulls with group mean
        df_grouped[target_col] = df_grouped[target_col].fillna(group_means)

        # Handle remaining nulls with overall mean/median
        if df_grouped[target_col].isnull().sum() > 0:
            if agg_func == "mean":
                fallback = df[target_col].mean()
            else:
                fallback = df[target_col].median()
            df_grouped[target_col] = df_grouped[target_col].fillna(fallback)

        return df_grouped

    def _apply_outlier_removal(
        self,
        df: pd.DataFrame,
        outlier_config: Dict[str, Any],
    ) -> pd.DataFrame:
        """Apply outlier detection and removal."""
        method = outlier_config.get("method")
        if not method:
            return df

        threshold = outlier_config.get("threshold", 1.5)
        df_filtered = df.copy()

        # Only numeric columns
        numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns

        if method == "IQR":
            df_filtered = self._remove_outliers_iqr(df_filtered, numeric_cols, threshold)
        elif method == "Zscore":
            df_filtered = self._remove_outliers_zscore(df_filtered, numeric_cols, threshold)

        self.log.append(
            {
                "task": "outlier_removal",
                "method": method,
                "original_rows": len(df),
                "filtered_rows": len(df_filtered),
                "removed": len(df) - len(df_filtered),
            }
        )

        return df_filtered

    def _remove_outliers_iqr(
        self,
        df: pd.DataFrame,
        cols: List[str],
        threshold: float = 1.5,
    ) -> pd.DataFrame:
        """Remove outliers using IQR method."""
        df_clean = df.copy()

        for col in cols:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            mask = (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
            df_clean = df_clean[mask]

        return df_clean

    def _remove_outliers_zscore(
        self,
        df: pd.DataFrame,
        cols: List[str],
        threshold: float = 3.0,
    ) -> pd.DataFrame:
        """
        Remove outliers using Z-score method.

        Args:
            df: DataFrame
            cols: Numeric columns to check
            threshold: Z-score threshold (default 3.0)

        Returns:
            DataFrame with outliers removed
        """
        df_clean = df.copy()

        for col in cols:
            # Calculate z-scores (only for non-null values)
            mask = df_clean[col].notna()
            valid_data = df_clean.loc[mask, col]

            if len(valid_data) > 0:
                z_scores = np.abs(stats.zscore(valid_data))
                # Keep values within threshold
                valid_indices = valid_data.index[z_scores <= threshold]
                df_clean = df_clean.loc[valid_indices]

        return df_clean

    def _cap_outliers(
        self,
        df: pd.DataFrame,
        cols: List[str],
        threshold: float = 1.5,
        method: str = "IQR",
    ) -> pd.DataFrame:
        """
        Cap outliers instead of removing them.

        Args:
            df: DataFrame
            cols: Columns to cap
            threshold: IQR multiplier or Z-score threshold
            method: 'IQR' or 'Zscore'

        Returns:
            DataFrame with capped outliers
        """
        df_capped = df.copy()

        for col in cols:
            if method == "IQR":
                Q1 = df_capped[col].quantile(0.25)
                Q3 = df_capped[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR
            else:  # Zscore
                mean = df_capped[col].mean()
                std = df_capped[col].std()
                lower = mean - threshold * std
                upper = mean + threshold * std

            # Cap values
            df_capped[col] = df_capped[col].clip(lower, upper)

            self.log.append(
                {
                    "task": "outlier_capping",
                    "column": col,
                    "method": method,
                    "lower_bound": lower,
                    "upper_bound": upper,
                }
            )

        return df_capped

    def encode_categorical(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: str = "label",
        drop_original: bool = False,
    ) -> pd.DataFrame:
        """
        Encode categorical variables.

        Args:
            df: DataFrame
            columns: Columns to encode
            method: 'label' or 'onehot'
            drop_original: Whether to drop original columns

        Returns:
            DataFrame with encoded columns
        """
        df_encoded = df.copy()

        if method == "label":
            for col in columns:
                if col in df_encoded.columns:
                    df_encoded[f"{col}_encoded"] = pd.factorize(df_encoded[col])[0]
                    if drop_original:
                        df_encoded = df_encoded.drop(columns=[col])
                    self.log.append({"task": "encoding", "column": col, "method": "label"})

        elif method == "onehot":
            df_encoded = pd.get_dummies(df_encoded, columns=columns, drop_first=False)
            self.log.append({"task": "encoding", "columns": columns, "method": "onehot"})

        return df_encoded

    def _save_transformation_log(self) -> None:
        """Save transformation log to JSON."""

        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            return obj

        log_entry = {
            "transformation_log": convert_numpy(self.log),
            "null_count_before": convert_numpy(self.null_count_before),
            "null_count_after": convert_numpy(self.null_count_after),
            "shape_before": list(self.shape_before),
            "shape_after": [convert_numpy(self.null_count_after), self.shape_before[1]],
            "null_values_removed": convert_numpy(self.null_count_before - self.null_count_after),
        }

        with open(self.output_dir / "transformation_log.json", "w") as f:
            json.dump(log_entry, f, indent=4)

    def get_log(self) -> List[Dict[str, Any]]:
        """Get transformation log."""
        return self.log

    def get_summary(self) -> Dict[str, Any]:
        """Get cleaning summary statistics."""
        return {
            "null_count_before": self.null_count_before,
            "null_count_after": self.null_count_after,
            "shape_before": self.shape_before,
            "transformations": len(self.log),
        }


class SklearnPipelineWrapper:
    """
    Wrapper for scikit-learn Pipeline integration.
    Provides sklearn-compatible preprocessing pipeline.
    """

    def __init__(self):
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler, LabelEncoder as SKLabelEncoder
        from sklearn.impute import SimpleImputer

        self.pipeline = None
        self.column_transformer = None

    def create_preprocessing_pipeline(
        self,
        numeric_cols: List[str],
        categorical_cols: List[str],
    ) -> "ColumnTransformer":
        """
        Create sklearn ColumnTransformer for preprocessing.

        Args:
            numeric_cols: List of numeric column names
            categorical_cols: List of categorical column names

        Returns:
            Configured ColumnTransformer
        """
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.impute import SimpleImputer

        # Numeric transformer: impute + scale
        numeric_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
        )

        # Categorical transformer: impute + encode
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        # Combined transformer
        self.column_transformer = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_cols),
                ("cat", categorical_transformer, categorical_cols),
            ]
        )

        return self.column_transformer

    def fit_transform(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str],
        categorical_cols: List[str],
    ) -> np.ndarray:
        """
        Fit and transform data using sklearn pipeline.

        Args:
            df: Input DataFrame
            numeric_cols: Numeric columns
            categorical_cols: Categorical columns

        Returns:
            Transformed numpy array
        """
        if not self.column_transformer:
            self.create_preprocessing_pipeline(numeric_cols, categorical_cols)

        return self.column_transformer.fit_transform(df)
