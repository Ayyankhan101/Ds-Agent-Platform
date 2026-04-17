import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class EDAAgent:
    """
    Agent for Deep Exploratory Data Analysis and Visualization.
    """

    def __init__(self, output_dir: str = "data/reports"):
        self.output_dir = output_dir

    def analyze(self, df: pd.DataFrame, target: str, correlation_method: str = "pearson"):
        """
        Generates core EDA metrics and plots.
        """
        results = {}

        # 1. Correlation Analysis
        numeric_df = df.select_dtypes(include=["number"])
        corr_matrix = numeric_df.corr(method=correlation_method)
        results["correlation"] = corr_matrix.to_dict()

        # 2. Basic Stats
        results["summary"] = df.describe().to_dict()

        # 3. Distribution Check for Target
        if target in df.columns:
            results["target_info"] = {
                "skew": df[target].skew() if pd.api.types.is_numeric_dtype(df[target]) else "N/A",
                "unique_values": df[target].nunique(),
            }

        return results

    def plot_correlations(self, df: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(10, 8))
        numeric_df = df.select_dtypes(include=["number"])
        sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        return fig
