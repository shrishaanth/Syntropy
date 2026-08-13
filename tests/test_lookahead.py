import pandas as pd
import numpy as np
import pytest
import hashlib
import json

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.backtest import Strategy
from src.metrics import calculate_metrics
from src.reporting import save_artifacts
from config import Config


def _make_synthetic_with_spike():
    """Create synthetic prices with a future spike."""
    np.random.seed(123)
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    prices = pd.DataFrame(
        np.random.randn(500, 3).cumsum(axis=0) + 100,
        index=dates,
        columns=["A", "B", "C"],
    )
    spike_idx = 400
    prices.iloc[spike_idx:, 0] += np.arange(0, 500 - spike_idx) * 0.5
    return prices


def test_no_lookahead_guarantee():
    """Backtest up to pre-spike date must yield identical results whether spike exists or not."""
    prices_with_spike = _make_synthetic_with_spike()
    prices_without_spike = prices_with_spike.copy()
    spike_idx = 400
    prices_without_spike.iloc[spike_idx:, 0] = prices_without_spike.iloc[spike_idx - 1, 0]

    cfg = Config()
    cutoff = prices_with_spike.index[350]

    strat = Strategy(train_window=60, test_window=20)
    results_with, _ = strat.run(prices_with_spike.loc[:cutoff])
    results_without, _ = strat.run(prices_without_spike.loc[:cutoff])

    pd.testing.assert_series_equal(
        results_with["Strategy"].reset_index(drop=True),
        results_without["Strategy"].reset_index(drop=True),
        atol=1e-10,
    )


def test_reproducibility():
    """Same config + same data produces identical metrics."""
    prices = _make_synthetic_with_spike()
    cfg = Config()

    strat = Strategy(train_window=60, test_window=20, transaction_cost=0.001)
    results1, weights1 = strat.run(prices)
    metrics1 = calculate_metrics(
        results1["Strategy"].pct_change().dropna(),
        {},
        weights1,
        cfg,
    )

    strat2 = Strategy(train_window=60, test_window=20, transaction_cost=0.001)
    results2, weights2 = strat2.run(prices)
    metrics2 = calculate_metrics(
        results2["Strategy"].pct_change().dropna(),
        {},
        weights2,
        cfg,
    )

    assert results1["Strategy"].equals(results2["Strategy"])
    assert weights1.equals(weights2)
    assert json.dumps(metrics1, sort_keys=True) == json.dumps(metrics2, sort_keys=True)
