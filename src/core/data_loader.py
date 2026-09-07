import pandas as pd
import yfinance as yf
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class DataLoaderConfig:
    symbols: List[str]
    start_date: str
    end_date: str
    price_col: str = 'Adj Close'

def fetch_data(config: DataLoaderConfig) -> Optional[pd.DataFrame]:
    try:
        data = yf.download(config.symbols, start=config.start_date, end=config.end_date)
        if data.empty:
            print("Warning: No data returned from yfinance.")
            return None

        prices = data[config.price_col]

        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=config.symbols[0])

        return prices.rename(columns={col: col.upper() for col in prices.columns})

    except Exception as e:
        print(f"Error fetching data: {e}")
        return None
