from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Config:
    symbols: tuple[str, ...] = ("AAPL", "MSFT", "NVDA", "JPM", "XOM")
    start_date: date = date(2020, 1, 1)
    end_date: date = date(2025, 1, 1)
    ewma_span: int = 60
    corr_span: int = 60
    train_years: int = 2
    test_months: int = 1
    step_months: int = 1
    execution_lag: int = 1
    transaction_cost_bps: float = 10.0
    max_asset_weight: float = 0.30
    min_asset_weight: float = 0.02
    risk_free_rate_annual: float = 0.04
