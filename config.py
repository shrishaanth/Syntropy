from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Config:
    symbols: tuple[str, ...] = (
        "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS",
        "BAJFINANCE.NS", "BAJAJFINSV.NS", "INDUSINDBK.NS", "SHRIRAMFIN.NS",
        "TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS",
        "RELIANCE.NS", "ONGC.NS", "COALINDIA.NS", "NTPC.NS", "POWERGRID.NS",
        "TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "GRASIM.NS", "ULTRACEMCO.NS",
        "ADANIENT.NS", "ADANIPORTS.NS",
        "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "TATACONSUM.NS", "ASIANPAINT.NS",
        "TITAN.NS", "TRENT.NS", "BRITANNIA.NS",
        "MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS",
        "HEROMOTOCO.NS",
        "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "APOLLOHOSP.NS",
        "BHARTIARTL.NS", "LT.NS", "BEL.NS", "UPL.NS",
        "GOLDBEES.NS",
    )
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
    risk_free_rate_annual: float = 0.06
    vol_target_annual: float = 0.10
    max_leverage: float = 2.0
    rebalance_band: float = 0.10
    weight_smoothing: float = 0.25
    leverage_smoothing: float = 1.0
