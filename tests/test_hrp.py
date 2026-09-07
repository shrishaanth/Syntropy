import pandas as pd
import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.hrp import allocate, _cluster_var, _clip_constraints
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


def test_cluster_var_uses_inverse_variance_weights():
    cov = np.diag([1.0, 100.0])
    assert _cluster_var(cov, [0, 1]) < 2.0


def test_allocate_tilts_away_from_riskier_asset():
    names = ["A", "B", "C", "D", "E"]
    cov = pd.DataFrame(np.diag([0.04, 0.04, 0.04, 0.04, 0.25]), index=names, columns=names)
    corr = pd.DataFrame(np.eye(5), index=names, columns=names)
    cfg = Config()
    w = allocate(cov, corr, cfg)
    assert abs(sum(w.values()) - 1.0) < 1e-6
    assert all(cfg.min_asset_weight - 1e-8 <= v <= cfg.max_asset_weight + 1e-8 for v in w.values())
    assert w["E"] < 1.0 / len(names)
    assert w["E"] < w["A"]


def test_clip_constraints_caps_concentrated_weight():
    w = _clip_constraints({"A": 0.70, "B": 0.20, "C": 0.06, "D": 0.04}, min_w=0.02, max_w=0.30)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert all(0.02 - 1e-9 <= v <= 0.30 + 1e-9 for v in w.values())
    assert w["A"] == pytest.approx(0.30)


def test_clip_constraints_lifts_tiny_weight():
    w = _clip_constraints({"A": 0.005, "B": 0.60, "C": 0.395}, min_w=0.02, max_w=0.50)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert all(0.02 - 1e-9 <= v <= 0.50 + 1e-9 for v in w.values())
    assert w["A"] > 0.005


def test_clip_constraints_infeasible_box_still_normalises():
    w = _clip_constraints({"A": 0.5, "B": 0.3, "C": 0.2}, min_w=0.02, max_w=0.30)
    assert abs(sum(w.values()) - 1.0) < 1e-9
