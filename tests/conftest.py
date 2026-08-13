import pandas as pd
import numpy as np
from datetime import date
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import Config


@pytest.fixture
def config():
    return Config()


@pytest.fixture
def prices():
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    data = {
        "A": np.linspace(100, 105, 10),
        "B": np.linspace(100, 110, 10),
        "C": [100 + (i % 2) * 2 for i in range(10)],
    }
    return pd.DataFrame(data, index=dates)
