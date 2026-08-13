# src/core/backtest.py
import pandas as pd
import numpy as np
from typing import List, Tuple
from .hrp import HierarchicalRiskParity
from .utils import clean_data, to_log_returns

class Strategy:
    """
    Implements a walk-forward backtesting engine for a quantitative portfolio strategy.
    """

    def __init__(self,
                 train_window: int = 252, # Approx 1 year of trading days
                 test_window: int = 63,   # Approx 3 months (quarterly rebalancing)
                 transaction_cost: float = 0.0005, # 5 bps
                 benchmarks: List[str] = ['SPY']):
        """
        Initializes the backtesting strategy.

        Args:
            train_window: Number of days in the training window for risk estimation.
            test_window: Number of days in the testing window for each rebalancing period.
            transaction_cost: Estimated cost per trade as a fraction of the trade value.
            benchmarks: List of benchmark ticker symbols.
        """
        self.train_window = train_window
        self.test_window = test_window
        self.transaction_cost = transaction_cost
        self.benchmarks = benchmarks

    def run(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Runs the walk-forward backtest.

        Args:
            data: DataFrame of prices for the assets (including benchmarks).

        Returns:
            A tuple (backtest_results, weight_history) where:
            - backtest_results: DataFrame with daily returns for the strategy and benchmarks.
            - weight_history: DataFrame tracking portfolio weights over time.
        """
        # 1. Prepare data (clean and convert to returns)
        asset_prices = data.drop(columns=self.benchmarks)
        benchmark_prices = data[self.benchmarks]

        returns = to_log_returns(asset_prices)
        benchmark_returns = to_log_returns(benchmark_prices)

        n_steps = len(returns)
        start_idx = self.train_window

        # Containers for results
        strategy_returns = []
        weight_history = []
        current_weights = None

        # 2. Walk-forward logic
        for i in range(start_idx, n_steps, self.test_window):
            # Define current train/test windows
            train_data = returns.iloc[i - self.train_window : i]
            test_end_idx = min(i + self.test_window, n_steps)
            test_data = returns.iloc[i : test_end_idx]

            # A. Rebalance: Compute new weights using HRP on training data
            optimizer = HierarchicalRiskParity()
            new_weights, _ = optimizer.optimize_portfolio(train_data)

            # B. Apply transaction costs on the rebalance
            cost_drag = 0
            if current_weights is not None:
                # Delta weight is the change in position
                delta_weights = np.abs(new_weights - current_weights)
                cost_drag = np.sum(delta_weights) * self.transaction_cost

            current_weights = new_weights

            # C. Record weights
            rebalance_date = returns.index[i]
            weight_history.append(pd.Series(current_weights, index=asset_prices.columns, name=rebalance_date))

            # D. Compute test returns (out-of-sample)
            # Strategy return = dot product of weights and returns
            period_returns = test_data.dot(current_weights)

            # Subtract cost drag from the first return of the test period
            if len(period_returns) > 0:
                period_returns.iloc[0] -= cost_drag

            strategy_returns.append(period_returns)

        # 3. Concatenate and finalize results
        strategy_series = pd.concat(strategy_returns)
        strategy_series.name = 'Strategy'

        # Sync benchmark returns to match the backtest period
        benchmark_series = benchmark_returns.reindex(strategy_series.index)

        # Combine results
        results = pd.concat([strategy_series, benchmark_series], axis=1)
        weights_df = pd.DataFrame(weight_history)

        return results, weights_df