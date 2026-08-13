# src/core/backtest.py
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional

from .utils import to_log_returns
from ..features import ewma_vol, ewmc_corr
from ..covariance import build_covariance, psd_repair
from ..hrp import allocate
from config import Config


class Strategy:
    """
    Implements a walk-forward backtesting engine for a quantitative portfolio strategy.
    """

    def __init__(self,
                 train_window: int = 252,
                 test_window: int = 63,
                 transaction_cost: float = 0.0005,
                 benchmarks: List[str] = None):
        self.train_window = train_window
        self.test_window = test_window
        self.transaction_cost = transaction_cost
        self.benchmarks = benchmarks or ['SPY']
        self.config = Config()

    def run(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        asset_prices = data.drop(columns=self.benchmarks, errors='ignore')
        benchmark_prices = data[self.benchmarks] if all(b in data.columns for b in self.benchmarks) else None

        returns_df = to_log_returns(asset_prices)
        benchmark_returns = to_log_returns(benchmark_prices) if benchmark_prices is not None else pd.DataFrame()

        n_steps = len(returns_df)
        start_idx = self.train_window

        strategy_returns = []
        weight_history = []
        cost_history = []
        current_weights = None

        for i in range(start_idx, n_steps, self.test_window):
            train_data = returns_df.iloc[i - self.train_window : i]
            test_end_idx = min(i + self.test_window, n_steps)
            test_data = returns_df.iloc[i : test_end_idx]

            vol = ewma_vol(train_data, span=self.config.ewma_span)
            corr = ewmc_corr(train_data, span=self.config.corr_span)
            cov = build_covariance(vol, corr)
            cov = psd_repair(cov)

            new_weights = pd.Series(allocate(cov, corr, self.config))

            cost_drag = 0
            if current_weights is not None:
                delta_weights = np.abs(new_weights - current_weights)
                cost_drag = np.sum(delta_weights) * self.transaction_cost

            current_weights = new_weights

            rebalance_date = returns_df.index[i]
            weight_history.append(pd.Series(current_weights, index=asset_prices.columns, name=rebalance_date))

            period_returns = test_data.dot(pd.Series(current_weights))

            if len(period_returns) > 0:
                period_returns.iloc[0] -= cost_drag

            strategy_returns.append(period_returns)
            cost_history.append(cost_drag)

        strategy_series = pd.concat(strategy_returns)
        strategy_series.name = 'Strategy'

        if not benchmark_returns.empty:
            benchmark_series = benchmark_returns.reindex(strategy_series.index)
            results = pd.concat([strategy_series, benchmark_series], axis=1)
        else:
            results = pd.DataFrame({'Strategy': strategy_series})

        weights_df = pd.DataFrame(weight_history)
        costs = pd.Series(cost_history, index=[w.name for w in weight_history])
        return results, weights_df, costs
