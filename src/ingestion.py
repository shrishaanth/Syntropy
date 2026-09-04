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


def _validate_and_clean(
    df: pd.DataFrame,
    min_coverage: float = 0.98,
    start_grace_days: int = 25,
    min_symbols: int = 5,
) -> pd.DataFrame:
    """Drop tickers without usable history over the window, then validate the rest.

    A large candidate universe (e.g. the NIFTY 50) will always contain names that
    listed after the start date or have gappy data on Yahoo Finance. Rather than
    failing the whole run, drop those names, align the survivors, and only then
    enforce the hard invariants.
    """
    n_rows = len(df)
    coverage = df.notna().mean()

    keep, dropped = [], {}
    for col in df.columns:
        first = df[col].first_valid_index()
        if coverage[col] < min_coverage:
            dropped[col] = f"{coverage[col]:.0%} coverage"
        elif first is None or df.index.get_loc(first) > start_grace_days:
            when = "no data" if first is None else f"starts {first.date()}"
            dropped[col] = when
        else:
            keep.append(col)

    if dropped:
        print(
            f"  dropped {len(dropped)}/{len(df.columns)} tickers "
            f"({n_rows} trading days requested): "
            + ", ".join(f"{k} [{v}]" for k, v in dropped.items())
        )
    df = df[keep]

    if len(df.columns) < min_symbols:
        raise ValueError(
            f"Only {len(df.columns)} tickers have usable history; need >= {min_symbols}"
        )

    dropped_rows = len(df)
    df = df.dropna(axis=0)
    if len(df) < dropped_rows:
        print(f"  dropped {dropped_rows - len(df)} rows with residual gaps")

    if (df <= 0).any().any():
        raise ValueError("Negative or zero prices detected")

    if not df.index.is_monotonic_increasing:
        raise ValueError("Date index is not monotonic")

    if df.index.has_duplicates:
        raise ValueError("Duplicate dates detected")

    return df
