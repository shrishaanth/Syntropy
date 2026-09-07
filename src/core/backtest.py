import pandas as pd
import numpy as np
from typing import List, Tuple, Optional

from .utils import to_log_returns
from ..features import ewma_vol, ewmc_corr
from ..covariance import build_covariance, psd_repair, shrink_correlation
from ..hrp import allocate
from config import Config


class Strategy:

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

        cfg = self.config
        rf_daily = cfg.risk_free_rate_annual / 252
        vol_target = cfg.vol_target_annual
        max_leverage = cfg.max_leverage
        band = cfg.rebalance_band
        weight_smoothing = cfg.weight_smoothing
        leverage_smoothing = cfg.leverage_smoothing

        strategy_returns = []
        weight_history = []
        cost_history = []
        hrp_weights = None
        traded_weights = None
        hrp_target = None
        leverage_state = None

        for i in range(start_idx, n_steps, self.test_window):
            train_data = returns_df.iloc[i - self.train_window : i]
            test_end_idx = min(i + self.test_window, n_steps)
            test_data = returns_df.iloc[i : test_end_idx]

            vol = ewma_vol(train_data, span=cfg.ewma_span)
            corr = ewmc_corr(train_data, span=cfg.corr_span)
            corr = shrink_correlation(corr, cfg.corr_shrinkage)
            cov = build_covariance(vol, corr)
            cov = psd_repair(cov)

            candidate_hrp = pd.Series(
                allocate(cov, corr, cfg), index=asset_prices.columns
            )

            if hrp_target is None:
                hrp_target = candidate_hrp
            else:
                hrp_target = (1.0 - weight_smoothing) * hrp_target + weight_smoothing * candidate_hrp
                hrp_target = hrp_target / hrp_target.sum()

            leverage_raw = 1.0
            if vol_target and vol_target > 0:
                port_vol = float(
                    np.sqrt(hrp_target.values @ cov.values @ hrp_target.values)
                )
                if port_vol > 0:
                    leverage_raw = min(vol_target / port_vol, max_leverage)

            if leverage_state is None:
                leverage_state = leverage_raw
            else:
                leverage_state = (
                    (1.0 - leverage_smoothing) * leverage_state
                    + leverage_smoothing * leverage_raw
                )

            candidate_traded = hrp_target * leverage_state

            if traded_weights is None:
                adopt = True
            else:
                adopt = float(np.abs(candidate_traded - traded_weights).sum()) > band

            cost_drag = 0.0
            if adopt:
                if traded_weights is not None:
                    cost_drag = float(
                        np.abs(candidate_traded - traded_weights).sum()
                    ) * self.transaction_cost
                hrp_weights = hrp_target
                traded_weights = candidate_traded

            rebalance_date = returns_df.index[i]
            weight_history.append(pd.Series(hrp_weights, index=asset_prices.columns, name=rebalance_date))

            cash_weight = 1.0 - float(traded_weights.sum())
            period_returns = test_data.dot(traded_weights) + cash_weight * rf_daily

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
