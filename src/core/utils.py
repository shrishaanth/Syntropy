import pandas as pd
import numpy as np

def clean_data(df: pd.DataFrame, dropna_thresh: float = 0.95) -> pd.DataFrame:
    cleaned_df = df.dropna(axis=1, thresh=int(dropna_thresh * len(df)))

    if cleaned_df.isnull().values.any():
        cleaned_df = cleaned_df.ffill().bfill()

    cleaned_df = cleaned_df.dropna(axis=0)

    return cleaned_df

def to_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    return np.log(df / df.shift(1)).dropna()
