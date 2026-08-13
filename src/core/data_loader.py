# src/core/data_loader.py
import pandas as pd
import yfinance as yf
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class DataLoaderConfig:
    """Configuration for the DataLoader."""
    symbols: List[str]
    start_date: str
    end_date: str
    price_col: str = 'Adj Close'

def fetch_data(config: DataLoaderConfig) -> Optional[pd.DataFrame]:
    """
    Fetches historical price data from Yahoo Finance.

    Args:
        config: A DataLoaderConfig object.

    Returns:
        A pandas DataFrame with the adjusted close prices, or None on failure.
    """
    try:
        data = yf.download(config.symbols, start=config.start_date, end=config.end_date)
        if data.empty:
            print("Warning: No data returned from yfinance.")
            return None

        prices = data[config.price_col]

        # Handle cases where only one symbol is fetched
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=config.symbols[0])

        return prices.rename(columns={col: col.upper() for col in prices.columns})

    except Exception as e:
        print(f"Error fetching data: {e}")
        return None
