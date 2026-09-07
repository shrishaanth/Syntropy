import numpy as np
import pandas as pd


def returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna()


def ewma_vol(returns: pd.DataFrame, span: int) -> pd.Series:
    daily_vol = returns.ewm(span=span, adjust=False).std().iloc[-1]
    return daily_vol * np.sqrt(252)


def ewmc_corr(returns: pd.DataFrame, span: int) -> pd.DataFrame:
    corr = returns.ewm(span=span, adjust=False).corr().iloc[-len(returns.columns) :]
    corr.index = returns.columns
    corr.columns = returns.columns
    corr = corr.clip(-1.0, 1.0)
    np.fill_diagonal(corr.values, 1.0)
    return corr
