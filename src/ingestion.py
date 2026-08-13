import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from config import Config


def download_and_save(config: Config) -> Path:
    """
    Download market data and save to parquet.

    Args:
        config: Config object with symbols and date range.

    Returns:
        Path to saved parquet file.
    """
    import yfinance as yf

    raw_path = Path("data/raw") / f"prices_{datetime.now().strftime('%Y%m%d')}.parquet"
    processed_path = Path("data/processed/prices.parquet")

    data = yf.download(
        list(config.symbols),
        start=config.start_date.isoformat(),
        end=config.end_date.isoformat(),
        auto_adjust=True,
        progress=False,
    )

    if data.empty:
        raise ValueError(f"No data returned from yfinance for symbols: {config.symbols}")

    prices = data["Close"]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=config.symbols[0])

    prices = prices.rename(columns={col: col.upper() for col in prices.columns})

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(raw_path)

    cleaned = _validate_and_clean(prices)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(processed_path)

    return processed_path


def _validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean price data."""
    if df.isna().any().any():
        missing = df.columns[df.isna().any()].tolist()
        raise ValueError(f"Missing values detected in symbols: {missing}")

    if (df <= 0).any().any():
        raise ValueError("Negative or zero prices detected")

    if not df.index.is_monotonic_increasing:
        raise ValueError("Date index is not monotonic")

    if df.index.has_duplicates:
        raise ValueError("Duplicate dates detected")

    return df
