from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Config:
    # Diversified cross-sector / cross-asset-class universe so the HRP clustering
    # has real structure to exploit (equities by sector, plus treasuries, gold,
    # and REITs).
    symbols: tuple[str, ...] = (
        "AAPL", "MSFT", "NVDA", "GOOGL",   # technology
        "JPM", "GS",                        # financials
        "XOM", "CVX",                       # energy
        "JNJ", "UNH",                       # health care
        "PG", "KO",                         # consumer staples
        "CAT",                              # industrials
        "NEE",                              # utilities
        "TLT", "IEF",                       # US treasuries (long / intermediate)
        "GLD",                              # gold
        "VNQ",                              # REITs
    )
    # A long sample so the walk-forward test spans several regimes (2018 selloff,
    # COVID crash, 2022 rate shock, recoveries) rather than a single window.
    start_date: date = date(2015, 1, 1)
    end_date: date = date(2025, 1, 1)
    ewma_span: int = 60
    corr_span: int = 60
    corr_shrinkage: float = 0.10
    train_years: int = 2
    test_months: int = 1
    step_months: int = 1
    execution_lag: int = 1
    transaction_cost_bps: float = 10.0
    max_asset_weight: float = 0.30
    min_asset_weight: float = 0.02
    risk_free_rate_annual: float = 0.04
    # Volatility-target overlay: scale total exposure toward this annualised vol,
    # capped at max_leverage. Set vol_target_annual = 0 to disable.
    vol_target_annual: float = 0.10
    max_leverage: float = 2.0
    # Banded rebalancing: only trade when the target book has drifted by more than
    # this (sum of absolute weight changes). Set to 0 to rebalance every period.
    rebalance_band: float = 0.10
    # EMA blend factors (weight on the newly computed value each period). 1.0 =
    # no smoothing; lower = stickier. Smoothing the allocation cuts turnover from
    # a noisy covariance estimate; the leverage is left unsmoothed so it can
    # de-risk quickly when volatility spikes.
    weight_smoothing: float = 0.25
    leverage_smoothing: float = 1.0
