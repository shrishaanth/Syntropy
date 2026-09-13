import pandas as pd
import numpy as np
from typing import Dict


def equal_weight(returns: pd.DataFrame, config) -> Dict[str, float]:
    n = len(returns.columns)
    return {col: 1.0 / n for col in returns.columns}


def inverse_variance(returns: pd.DataFrame, config) -> Dict[str, float]:
    var = returns.var().replace(0, 1e-10)
    inv_var = 1.0 / var
    weights = inv_var / inv_var.sum()
    return weights.to_dict()


def benchmark_returns(log_returns: pd.DataFrame, config) -> Dict[str, pd.Series]:
    """Equal-weight and inverse-variance return series for the same universe,
    used both by the pipeline script and the dashboard's charts."""
    return {
        "equal_weight": log_returns.dot(pd.Series(equal_weight(log_returns, config))),
        "inverse_variance": log_returns.dot(pd.Series(inverse_variance(log_returns, config))),
    }
