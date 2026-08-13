import pandas as pd
import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.hrp import allocate
from src.features import ewma_vol, ewmc_corr
from src.covariance import build_covariance, psd_repair
from config import Config


def test_weights_sum_to_one():
    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    np.random.seed(0)
    returns_df = pd.DataFrame(
        np.random.randn(60, 5),
        index=dates,
        columns=["AAPL", "MSFT", "NVDA", "JPM", "XOM"],
    )
    cfg = Config()
    vol = ewma_vol(returns_df, cfg.ewma_span)
    corr = ewmc_corr(returns_df, cfg.corr_span)
    cov = build_covariance(vol, corr)
    cov = psd_repair(cov)
    w = allocate(cov, corr, cfg)
    assert abs(sum(w.values()) - 1.0) < 1e-6
    assert all(v >= cfg.min_asset_weight - 1e-8 for v in w.values())
    assert all(v <= cfg.max_asset_weight + 1e-8 for v in w.values())
