import pandas as pd
import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.benchmarks import equal_weight, inverse_variance
from config import Config


def test_equal_weight():
    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    returns_df = pd.DataFrame(
        np.random.randn(60, 4),
        index=dates,
        columns=["A", "B", "C", "D"],
    )
    cfg = Config()
    w = equal_weight(returns_df, cfg)
    assert abs(sum(w.values()) - 1.0) < 1e-6
    expected = 1.0 / 4
    assert all(abs(v - expected) < 1e-6 for v in w.values())


def test_inverse_variance():
    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    np.random.seed(1)
    returns_df = pd.DataFrame(
        np.random.randn(60, 3),
        index=dates,
        columns=["A", "B", "C"],
    )
    cfg = Config()
    w = inverse_variance(returns_df, cfg)
    assert abs(sum(w.values()) - 1.0) < 1e-6
    var = returns_df.var()
    inv_var = (1.0 / var).to_dict()
    expected = {k: v / sum(inv_var.values()) for k, v in inv_var.items()}
    for k in expected:
        assert np.isclose(w[k], expected[k], atol=1e-6)
