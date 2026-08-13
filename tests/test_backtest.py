import pandas as pd
import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.backtest import Strategy


def test_walk_forward_no_nan():
    dates = pd.date_range("2024-01-01", periods=252, freq="B")
    np.random.seed(42)
    prices = pd.DataFrame(
        np.random.randn(252, 3).cumsum(axis=0) + 100,
        index=dates,
        columns=["A", "B", "C"],
    )
    strat = Strategy(train_window=60, test_window=20)
    results, weights_df = strat.run(prices)
    assert not results.isna().any().any()
    assert len(results) > 0
    assert len(weights_df) > 0


def test_weights_sum_to_one():
    dates = pd.date_range("2024-01-01", periods=252, freq="B")
    np.random.seed(42)
    prices = pd.DataFrame(
        np.random.randn(252, 3).cumsum(axis=0) + 100,
        index=dates,
        columns=["A", "B", "C"],
    )
    strat = Strategy(train_window=60, test_window=20)
    _, weights_df = strat.run(prices)
    row_sums = weights_df.sum(axis=1)
    assert np.allclose(row_sums.values, 1.0, atol=1e-6)


def test_window_count():
    dates = pd.date_range("2024-01-01", periods=252, freq="B")
    np.random.seed(42)
    prices = pd.DataFrame(
        np.random.randn(252, 3).cumsum(axis=0) + 100,
        index=dates,
        columns=["A", "B", "C"],
    )
    strat = Strategy(train_window=60, test_window=20)
    _, weights_df = strat.run(prices)
    assert len(weights_df) == 10
