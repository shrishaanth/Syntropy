import pandas as pd
import numpy as np
from typing import Dict


def equal_weight(returns: pd.DataFrame, config) -> Dict[str, float]:
    """Equal-weight allocation."""
    n = len(returns.columns)
    return {col: 1.0 / n for col in returns.columns}


def inverse_variance(returns: pd.DataFrame, config) -> Dict[str, float]:
    """Inverse-variance allocation."""
    var = returns.var().replace(0, 1e-10)
    inv_var = 1.0 / var
    weights = inv_var / inv_var.sum()
    return weights.to_dict()
