import pandas as pd
import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.covariance import build_covariance, psd_repair


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
