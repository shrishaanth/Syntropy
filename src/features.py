import numpy as np
import pandas as pd


def returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily log returns from prices."""
    return np.log(prices / prices.shift(1)).dropna()


def ewma_vol(returns: pd.DataFrame, span: int) -> pd.Series:
    """Compute annualized EWMA volatility per asset."""
    daily_vol = returns.ewm(span=span, adjust=False).std().iloc[-1]
    return daily_vol * np.sqrt(252)


def ewmc_corr(returns: pd.DataFrame, span: int) -> pd.DataFrame:
    """Compute EWMA correlation matrix at the last timestamp."""
    corr = returns.ewm(span=span, adjust=False).corr().iloc[-len(returns.columns) :]
    corr.index = returns.columns
    corr.columns = returns.columns
    return corr
