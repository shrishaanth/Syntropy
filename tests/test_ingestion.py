import numpy as np
import pandas as pd
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ingestion import _validate_and_clean


def _frame(n=300):
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    rng = np.random.default_rng(0)
    cols = list("ABCDEF")
    base = 100 + rng.standard_normal((n, len(cols))).cumsum(axis=0)
    return pd.DataFrame(base + 50, index=idx, columns=cols)


def test_keeps_full_history_columns():
    df = _frame()
    out = _validate_and_clean(df)
    assert list(out.columns) == list("ABCDEF")
    assert len(out) == len(df)


def test_drops_late_listed_ticker():
    df = _frame()
    df.loc[df.index[:120], "C"] = np.nan
    out = _validate_and_clean(df, start_grace_days=25)
    assert "C" not in out.columns
    assert {"A", "B", "D", "E", "F"} <= set(out.columns)
    assert not out.isna().any().any()


def test_drops_gappy_ticker_below_coverage():
    df = _frame()
    rng = np.random.default_rng(1)
    gap_rows = rng.choice(len(df), size=int(0.1 * len(df)), replace=False)
    df.iloc[gap_rows, df.columns.get_loc("B")] = np.nan
    out = _validate_and_clean(df, min_coverage=0.98)
    assert "B" not in out.columns


def test_raises_when_too_few_survive():
    df = _frame()
    df.loc[df.index[:200], ["B", "C", "D", "E"]] = np.nan
    with pytest.raises(ValueError, match="usable history"):
        _validate_and_clean(df, min_symbols=3)


def test_rejects_nonpositive_prices():
    df = _frame()
    df.iloc[10, 0] = -1.0
    with pytest.raises(ValueError, match="Negative or zero"):
        _validate_and_clean(df)
