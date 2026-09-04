from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Config:
    # NIFTY 50 constituents (NSE, via Yahoo Finance ".NS" suffix) plus a gold ETF
    # as a non-equity diversifier. Names without usable price history over the
    # backtest window are dropped automatically during ingestion, so this list is
    # the *candidate* universe, not necessarily the traded one.
    symbols: tuple[str, ...] = (
        # financials
        "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS",
        "BAJFINANCE.NS", "BAJAJFINSV.NS", "INDUSINDBK.NS", "SHRIRAMFIN.NS",
        # IT
        "TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS",
        # energy / materials / utilities
        "RELIANCE.NS", "ONGC.NS", "COALINDIA.NS", "NTPC.NS", "POWERGRID.NS",
        "TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "GRASIM.NS", "ULTRACEMCO.NS",
        "ADANIENT.NS", "ADANIPORTS.NS",
        # consumer
        "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "TATACONSUM.NS", "ASIANPAINT.NS",
        "TITAN.NS", "TRENT.NS", "BRITANNIA.NS",
        # autos
        "MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS",
        "HEROMOTOCO.NS",
        # pharma / health care
        "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "APOLLOHOSP.NS",
        # telecom / industrials / other
        "BHARTIARTL.NS", "LT.NS", "BEL.NS", "UPL.NS",
        # non-equity diversifier
        "GOLDBEES.NS",  # Nippon India ETF Gold BeES
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
    risk_free_rate_annual: float = 0.06  # ~average Indian 91-day T-bill yield, 2015-2024
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
