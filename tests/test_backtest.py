import pandas as pd
import numpy as np
import pytest

import sys
import os
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.backtest import Strategy
from src.metrics import annualized_volatility


def _synthetic_prices(n=252, k=3, seed=42):
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.standard_normal((n, k)).cumsum(axis=0) + 100,
        index=dates,
        columns=[chr(ord("A") + i) for i in range(k)],
    )


def test_walk_forward_no_nan():
    dates = pd.date_range("2024-01-01", periods=252, freq="B")
    np.random.seed(42)
    prices = pd.DataFrame(
        np.random.randn(252, 3).cumsum(axis=0) + 100,
        index=dates,
        columns=["A", "B", "C"],
    )
    strat = Strategy(train_window=60, test_window=20)
    results, weights_df, costs = strat.run(prices)
    assert not results.isna().any().any()
    assert len(results) > 0
    assert len(weights_df) > 0
    assert len(costs) == len(weights_df)


def test_weights_sum_to_one():
    dates = pd.date_range("2024-01-01", periods=252, freq="B")
    np.random.seed(42)
    prices = pd.DataFrame(
        np.random.randn(252, 3).cumsum(axis=0) + 100,
        index=dates,
        columns=["A", "B", "C"],
    )
    strat = Strategy(train_window=60, test_window=20)
    _, weights_df, _ = strat.run(prices)
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
    _, weights_df, _ = strat.run(prices)
    assert len(weights_df) == 10


def test_vol_target_caps_exposure_and_reduces_vol():
    prices = _synthetic_prices(seed=7)

    # Low target + no leverage headroom -> the strategy must sit partly in cash
    # and realise a lower vol than the fully-invested (targeting-off) version.
    strat_off = Strategy(train_window=60, test_window=20)
    strat_off.config = replace(strat_off.config, vol_target_annual=0.0)
    res_off, _, _ = strat_off.run(prices)

    strat_on = Strategy(train_window=60, test_window=20)
    strat_on.config = replace(strat_on.config, vol_target_annual=0.05, max_leverage=1.0)
    res_on, _, _ = strat_on.run(prices)

    assert annualized_volatility(res_on["Strategy"]) < annualized_volatility(res_off["Strategy"])


def test_banded_rebalancing_cuts_turnover_and_cost():
    prices = _synthetic_prices(seed=11)

    strat_every = Strategy(train_window=60, test_window=20)
    strat_every.config = replace(strat_every.config, rebalance_band=0.0)
    _, w_every, costs_every = strat_every.run(prices)

    strat_banded = Strategy(train_window=60, test_window=20)
    strat_banded.config = replace(strat_banded.config, rebalance_band=0.5)
    _, w_banded, costs_banded = strat_banded.run(prices)

    # A wide band should suppress trading: lower cumulative cost and at least one
    # rebalance row carried forward unchanged.
    assert costs_banded.sum() <= costs_every.sum()
    assert (w_banded.diff().iloc[1:].abs().sum(axis=1) < 1e-12).any()
