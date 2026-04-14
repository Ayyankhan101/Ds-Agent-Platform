import pytest
import pandas as pd
import numpy as np
from ds_platform.agents.cleaning import CleaningPipeline

def test_cleaning_imputation():
    df = pd.DataFrame({'a': [1, 2, None, 4]})
    agent = CleaningPipeline(output_dir="data/outputs")
    cleaned_df = agent.clean(df, {'a': 'mean'}, {'method': None})
    
    assert cleaned_df['a'].isnull().sum() == 0
    assert cleaned_df['a'].iloc[2] == df['a'].mean()

def test_cleaning_outliers():
    # Large outlier
    df = pd.DataFrame({'a': [1, 2, 3, 4, 1000]})
    agent = CleaningPipeline(output_dir="data/outputs")
    cleaned_df = agent.clean(df, {}, {'method': 'IQR', 'threshold': 1.5})
    
    # 1000 should be removed
    assert len(cleaned_df) == 4
    assert 1000 not in cleaned_df['a'].values
