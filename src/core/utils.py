# src/core/utils.py
import pandas as pd
import numpy as np

def clean_data(df: pd.DataFrame, dropna_thresh: float = 0.95) -> pd.DataFrame:
    """
    Cleans the raw price data.

    - Drops columns with too many NaNs.
    - Forward-fills remaining NaNs.
    - Drops any remaining rows with NaNs.

    Args:
        df: DataFrame of raw prices.
        dropna_thresh: Threshold for dropping columns with NaNs.

    Returns:
        A cleaned pandas DataFrame.
    """
    # Drop assets with too much missing data
    cleaned_df = df.dropna(axis=1, thresh=int(dropna_thresh * len(df)))

    # Forward-fill and then back-fill any remaining NaNs
    if cleaned_df.isnull().values.any():
        cleaned_df = cleaned_df.ffill().bfill()

    # Drop any rows that still have NaNs (e.g., at the very beginning)
    cleaned_df = cleaned_df.dropna(axis=0)

    return cleaned_df

def to_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Converts a price series to a log returns series."""
    return np.log(df / df.shift(1)).dropna()
