"""
Deep Exploratory Data Analysis Agent.
Correlation analysis, distributions, skewness, violin plots, KDE, and group-based analysis.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path


class EDAAgent:
    """
    Agent for Deep Exploratory Data Analysis and Visualization.

    Features:
    - Pearson & Spearman correlation
    - Distribution analysis with skewness detection
    - Violin plots for categorical vs numeric
    - KDE plots for density estimation
    - Group-based statistical analysis
    """

    def __init__(self, output_dir: str = "data/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def analyze(
        self,
        df: pd.DataFrame,
        target: str,
        correlation_method: str = "pearson",
    ) -> Dict[str, Any]:
        """
        Generate core EDA metrics and analysis.

        Args:
            df: Input DataFrame
            target: Target variable name
            correlation_method: 'pearson' or 'spearman'

        Returns:
            Dictionary with analysis results
        """
        results = {}

        # 1. Correlation Analysis
        numeric_df = df.select_dtypes(include=["number"])
        corr_matrix = numeric_df.corr(method=correlation_method)
        results["correlation"] = corr_matrix.to_dict()
        results["correlation_method"] = correlation_method

        # Store for later use
        results["corr_matrix"] = corr_matrix

        # 2. Basic Stats
        results["summary"] = df.describe().to_dict()

        # 3. Distribution Check for Target
        if target in df.columns:
            target_data = df[target]
            if pd.api.types.is_numeric_dtype(target_data):
                skewness = target_data.skew()
                kurtosis = target_data.kurtosis()
                results["target_info"] = {
                    "skew": float(skewness),
                    "kurtosis": float(kurtosis),
                    "skew_interpretation": self._interpret_skewness(skewness),
                    "unique_values": int(target_data.nunique()),
                }
            else:
                results["target_info"] = {
                    "skew": "N/A",
                    "unique_values": int(target_data.nunique()),
                }

        return results

    def _interpret_skewness(self, skew: float) -> str:
        """
        Interpret skewness value.

        Args:
            skew: Skewness coefficient

        Returns:
            Interpretation string
        """
        if abs(skew) < 0.5:
            return "approximately symmetric"
        elif skew > 0.5 and skew < 1:
            return "moderately right-skewed"
        elif skew > 1:
            return "highly right-skewed"
        elif skew < -0.5 and skew > -1:
            return "moderately left-skewed"
        else:
            return "highly left-skewed"

    def plot_correlations(
        self,
        df: pd.DataFrame,
        method: str = "pearson",
    ) -> plt.Figure:
        """Plot correlation heatmap."""
        fig, ax = plt.subplots(figsize=(10, 8))
        numeric_df = df.select_dtypes(include=["number"])
        sns.heatmap(
            numeric_df.corr(method=method),
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            ax=ax,
            center=0,
        )
        ax.set_title(f"{method.title()} Correlation Matrix")
        return fig

    def plot_correlations_plotly(
        self,
        df: pd.DataFrame,
        method: str = "pearson",
    ) -> go.Figure:
        """Plot interactive correlation heatmap with Plotly."""
        numeric_df = df.select_dtypes(include=["number"])
        corr_matrix = numeric_df.corr(method=method)

        fig = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu",
            title=f"{method.title()} Correlation Matrix",
            aspect="auto",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
        )
        return fig

    def plot_violin(
        self,
        df: pd.DataFrame,
        numeric_col: str,
        categorical_col: str,
    ) -> go.Figure:
        """
        Create violin plot for numeric vs categorical variable.

        Args:
            df: DataFrame
            numeric_col: Numeric column
            categorical_col: Categorical column for grouping

        Returns:
            Plotly figure
        """
        fig = px.violin(
            df,
            y=numeric_col,
            x=categorical_col,
            box=True,
            points="outliers",
            title=f"Distribution of {numeric_col} by {categorical_col}",
            color=categorical_col,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            yaxis=dict(gridcolor="gray"),
        )
        return fig

    def plot_kde(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        fill: bool = True,
    ) -> plt.Figure:
        """
        Create KDE (Kernel Density Estimation) plots.

        Args:
            df: DataFrame
            columns: Columns to plot (default: all numeric)
            fill: Whether to fill the KDE curves

        Returns:
            Matplotlib figure
        """
        if columns is None:
            columns = df.select_dtypes(include=["number"]).columns.tolist()

        n_cols = len(columns)
        n_rows = (n_cols + 1) // 2

        fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]

        for i, col in enumerate(columns):
            ax = axes[i]
            # Drop NA values for KDE
            data = df[col].dropna()

            if len(data) > 1:
                sns.kdeplot(
                    data=data,
                    ax=ax,
                    fill=fill,
                    alpha=0.5,
                    color="steelblue",
                    lw=2,
                )
                ax.axvline(data.mean(), color="red", linestyle="--", label="Mean")
                ax.axvline(data.median(), color="green", linestyle=":", label="Median")
                ax.set_title(f"KDE: {col}")
                ax.legend()
                ax.grid(True, alpha=0.3)

        # Hide unused axes
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        return fig

    def plot_kde_plotly(
        self,
        df: pd.DataFrame,
        column: str,
    ) -> go.Figure:
        """
        Create interactive KDE-like density plot with Plotly.

        Args:
            df: DataFrame
            column: Column to plot

        Returns:
            Plotly figure
        """
        data = df[column].dropna()

        # Calculate KDE manually for density histogram
        fig = px.histogram(
            df,
            x=column,
            nbins=50,
            marginal="box",
            title=f"Distribution: {column}",
            color_discrete_sequence=["#00CC96"],
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
        )
        return fig

    def plot_group_analysis(
        self,
        df: pd.DataFrame,
        numeric_col: str,
        categorical_col: str,
    ) -> Tuple[pd.DataFrame, go.Figure]:
        """
        Perform group-based statistical analysis.

        Args:
            df: DataFrame
            numeric_col: Numeric column to analyze
            categorical_col: Column to group by

        Returns:
            Tuple of (grouped stats DataFrame, visualization)
        """
        # Group statistics
        grouped = (
            df.groupby(categorical_col)[numeric_col]
            .agg(["count", "mean", "std", "min", "median", "max"])
            .round(2)
        )

        # Visualization: boxplot
        fig = px.box(
            df,
            x=categorical_col,
            y=numeric_col,
            title=f"{numeric_col} by {categorical_col}",
            color=categorical_col,
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
        )

        return grouped, fig

    def plot_distribution_grid(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
    ) -> plt.Figure:
        """
        Create grid of distribution plots for all numeric columns.

        Args:
            df: DataFrame
            columns: Columns to plot (default: all numeric)

        Returns:
            Matplotlib figure
        """
        if columns is None:
            columns = df.select_dtypes(include=["number"]).columns.tolist()

        n_cols = len(columns)
        n_rows = (n_cols + 1) // 2

        fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4 * n_rows))
        if n_rows == 1:
            axes = [axes]
        axes = [a for ax in axes for a in (ax if hasattr(ax, "__iter__") else [ax])]

        for i, col in enumerate(columns[:n_cols]):
            if i < len(axes):
                ax = axes[i]
                data = df[col].dropna()

                # Histogram with KDE
                sns.histplot(data, ax=ax, kde=True, color="steelblue", alpha=0.6)
                ax.set_title(f"{col}")
                ax.axvline(data.mean(), color="red", linestyle="--", lw=1.5)
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def get_skewness_report(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate skewness report for all numeric columns.

        Args:
            df: DataFrame

        Returns:
            DataFrame with skewness metrics
        """
        numeric_df = df.select_dtypes(include=["number"])

        report = pd.DataFrame(
            {
                "skewness": numeric_df.skew(),
                "kurtosis": numeric_df.kurtosis(),
                "mean": numeric_df.mean(),
                "median": numeric_df.median(),
                "std": numeric_df.std(),
            }
        )
        report["interpretation"] = report["skewness"].apply(self._interpret_skewness)

        return report.sort_values("skewness", key=abs, ascending=False)

    def get_missing_report(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate missing values report.

        Args:
            df: DataFrame

        Returns:
            DataFrame with missing value analysis
        """
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)

        report = pd.DataFrame(
            {
                "missing_count": missing,
                "missing_pct": missing_pct,
                "dtype": df.dtypes,
            }
        )

        return report[report["missing_count"] > 0].sort_values("missing_count", ascending=False)

    def get_correlation_pairs(
        self,
        df: pd.DataFrame,
        method: str = "pearson",
        threshold: float = 0.5,
    ) -> pd.DataFrame:
        """
        Get correlation pairs above threshold.

        Args:
            df: DataFrame
            method: Correlation method
            threshold: Absolute correlation threshold

        Returns:
            DataFrame with correlation pairs
        """
        numeric_df = df.select_dtypes(include=["number"])
        corr_matrix = numeric_df.corr(method=method)

        pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) >= threshold:
                    pairs.append(
                        {
                            "variable_1": corr_matrix.columns[i],
                            "variable_2": corr_matrix.columns[j],
                            "correlation": round(corr_val, 4),
                        }
                    )

        return pd.DataFrame(pairs).sort_values("correlation", key=abs, ascending=False)
