import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.domain import Weights


def test_valid_weights():
    w = Weights(weights={"AAPL": 0.5, "MSFT": 0.5})
    assert sum(w.weights.values()) == 1.0


def test_invalid_weights_sum():
    with pytest.raises(ValueError):
        Weights(weights={"AAPL": 0.9, "MSFT": 0.2})
