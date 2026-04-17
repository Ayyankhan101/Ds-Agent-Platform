"""
Synthetic Data Generator for Data Science Platform.
Generates realistic sales/customer data with missing values and outliers.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path


def generate_sales_data(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic sales data with realistic distributions.

    Args:
        n: Number of records to generate
        seed: Random seed for reproducibility

    Returns:
        DataFrame with sales data
    """
    np.random.seed(seed)

    # Core numeric features
    age = np.random.randint(18, 75, n)
    # Income: log-normal for realistic skewed distribution
    income = np.random.lognormal(10.5, 0.8, n)
    # Spending score: uniform distribution 1-100
    spending_score = np.random.randint(1, 100, n)
    # Purchase amount: log-normal
    purchase_amount = np.random.lognormal(5, 1, n)

    # Add extreme outliers (~2% of data)
    outlier_idx = np.random.choice(n, int(n * 0.02), replace=False)
    income[outlier_idx] *= 10  # Extreme high income
    purchase_amount[outlier_idx] *= 5  # Extreme purchase

    # Categorical features
    categories = np.random.choice(
        ["Electronics", "Clothing", "Food", "Home", "Sports", "Books", "Toys"],
        n,
        p=[0.25, 0.20, 0.18, 0.15, 0.10, 0.07, 0.05],
    )
    regions = np.random.choice(["North", "South", "East", "West"], n)
    channels = np.random.choice(["Online", "Store", "Catalog"], n, p=[0.5, 0.35, 0.15])

    # Date features
    start_date = datetime(2023, 1, 1)
    purchase_date = [start_date + timedelta(days=np.random.randint(0, 365)) for _ in range(n)]

    # Binary target for classification: converted (1) or not (0)
    # Based on spending behavior
    converted = ((spending_score > 50) | (income > 30000)).astype(int)

    # Additional features for feature engineering
    num_purchases = np.random.poisson(5, n)
    tenure_days = np.random.randint(30, 2000, n)
    satisfaction_score = np.random.randint(1, 11, n)  # 1-10 scale

    df = pd.DataFrame(
        {
            "age": age,
            "income": income,
            "spending_score": spending_score,
            "purchase_amount": purchase_amount,
            "category": categories,
            "region": regions,
            "channel": channels,
            "purchase_date": purchase_date,
            "converted": converted,
            "num_purchases": num_purchases,
            "tenure_days": tenure_days,
            "satisfaction_score": satisfaction_score,
        }
    )

    # Introduce missing values (~5% per column)
    missing_cols = ["age", "income", "spending_score", "satisfaction_score"]
    for col in missing_cols:
        missing_idx = np.random.choice(n, int(n * 0.05), replace=False)
        df.loc[missing_idx, col] = np.nan

    return df


def generate_customer_segments(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Generate customer data segmented by behavior for group analysis.

    Args:
        n: Number of records
        seed: Random seed

    Returns:
        DataFrame with customer segments
    """
    np.random.seed(seed)

    # Create segments
    segments = np.random.choice(
        ["Premium", "Standard", "Basic", "Inactive"], n, p=[0.2, 0.4, 0.3, 0.1]
    )

    # Metrics correlated with segment
    data = []
    for seg in segments:
        if seg == "Premium":
            data.append(
                {
                    "segment": seg,
                    "annual_spend": np.random.normal(5000, 500),
                    "visits_per_month": np.random.poisson(10),
                    "items_per_order": np.random.poisson(5),
                    "support_tickets": np.random.poisson(1),
                }
            )
        elif seg == "Standard":
            data.append(
                {
                    "segment": seg,
                    "annual_spend": np.random.normal(1500, 300),
                    "visits_per_month": np.random.poisson(5),
                    "items_per_order": np.random.poisson(3),
                    "support_tickets": np.random.poisson(2),
                }
            )
        elif seg == "Basic":
            data.append(
                {
                    "segment": seg,
                    "annual_spend": np.random.normal(500, 150),
                    "visits_per_month": np.random.poisson(2),
                    "items_per_order": np.random.poisson(2),
                    "support_tickets": np.random.poisson(1),
                }
            )
        else:  # Inactive
            data.append(
                {
                    "segment": seg,
                    "annual_spend": np.random.normal(50, 20),
                    "visits_per_month": np.random.poisson(0),
                    "items_per_order": np.random.poisson(1),
                    "support_tickets": np.random.poisson(0),
                }
            )

    df = pd.DataFrame(data)

    # Add some missing values for realistic cleaning scenarios
    for col in ["annual_spend", "visits_per_month"]:
        missing_idx = np.random.choice(n, int(n * 0.03), replace=False)
        df.loc[missing_idx, col] = np.nan

    return df


def save_sample_data(output_dir: str = "data") -> None:
    """Generate and save all sample datasets."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Sales data
    sales_df = generate_sales_data(5000, seed=42)
    sales_df.to_csv(output_path / "sales_data.csv", index=False)
    print(f"✓ Generated sales_data.csv: {sales_df.shape}")

    # Customer segments
    segments_df = generate_customer_segments(2000, seed=42)
    segments_df.to_csv(output_path / "customer_segments.csv", index=False)
    print(f"✓ Generated customer_segments.csv: {segments_df.shape}")


if __name__ == "__main__":
    save_sample_data()
