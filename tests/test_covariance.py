import pandas as pd
import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.covariance import build_covariance, psd_repair, shrink_correlation


def test_build_covariance():
    vol = pd.Series([0.2, 0.3], index=["A", "B"])
    corr = pd.DataFrame([[1.0, 0.5], [0.5, 1.0]], index=["A", "B"], columns=["A", "B"])
    cov = build_covariance(vol, corr)
    expected = np.array([[0.04, 0.03], [0.03, 0.09]])
    np.testing.assert_allclose(cov.values, expected, atol=1e-8)


def test_psd_repair_valid():
    cov = pd.DataFrame(np.array([[1.0, 0.5], [0.5, 1.0]]), index=["A", "B"], columns=["A", "B"])
    repaired = psd_repair(cov)
    vals = np.linalg.eigvalsh(repaired.values)
    assert (vals >= -1e-8).all()


def test_shrink_correlation_zero_delta_is_identity():
    corr = pd.DataFrame(
        [[1.0, 0.8, 0.2], [0.8, 1.0, 0.3], [0.2, 0.3, 1.0]],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    np.testing.assert_allclose(shrink_correlation(corr, 0.0).values, corr.values)


def test_shrink_correlation_pulls_toward_mean():
    corr = pd.DataFrame(
        [[1.0, 0.8, 0.2], [0.8, 1.0, 0.3], [0.2, 0.3, 1.0]],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    mean_off = np.mean([0.8, 0.2, 0.3])
    shrunk = shrink_correlation(corr, 0.5)

    np.testing.assert_allclose(np.diag(shrunk.values), 1.0)
    np.testing.assert_allclose(shrunk.values, shrunk.values.T)
    assert shrunk.loc["A", "B"] == pytest.approx(0.5 * 0.8 + 0.5 * mean_off)
    assert abs(shrunk.loc["A", "B"] - mean_off) < abs(corr.loc["A", "B"] - mean_off)


def test_shrink_correlation_full_delta_is_target():
    corr = pd.DataFrame(
        [[1.0, 0.8, 0.2], [0.8, 1.0, 0.3], [0.2, 0.3, 1.0]],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    mean_off = np.mean([0.8, 0.2, 0.3])
    shrunk = shrink_correlation(corr, 1.0)
    off_diag = shrunk.values[~np.eye(3, dtype=bool)]
    np.testing.assert_allclose(off_diag, mean_off)
