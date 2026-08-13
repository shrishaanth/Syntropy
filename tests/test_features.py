import pandas as pd
import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.features import returns, ewma_vol, ewmc_corr


def test_returns_correctness(prices):
    ret = returns(prices)
    assert ret.shape == (9, 3)
    b_prices = prices["B"].values
    expected = np.log(b_prices[2] / b_prices[1])
    assert np.isclose(ret.iloc[1, 1], expected, atol=1e-6)


def test_ewma_vol_shape(prices):
    ret = returns(prices)
    vol = ewma_vol(ret, span=3)
    assert len(vol) == 3


def test_ewmc_corr_bounds(prices):
    ret = returns(prices)
    corr = ewmc_corr(ret, span=3)
    assert corr.shape == (3, 3)
    assert np.allclose(np.diag(corr), 1.0, atol=1e-6)
    assert (-1 <= corr.values).all() and (corr.values <= 1).all()
