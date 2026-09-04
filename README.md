# Syntropy

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Tests](https://img.shields.io/badge/tests-31%20passing-green)

**A point-in-time quantitative pipeline that estimates time-varying risk via EWMA, allocates capital across the NIFTY 50 through a from-scratch Hierarchical Risk Parity (HRP) engine with a volatility-target overlay, and validates performance via walk-forward backtesting with provable zero future-data leakage. Outputs an interactive Streamlit dashboard for strategy benchmarking against equal-weight and inverse-variance baselines.**

## What It Does

Syntropy replaces guesswork with a reproducible, mathematical portfolio allocation system:

1. **Downloads** split/dividend-adjusted daily prices from Yahoo Finance for the NIFTY 50 constituents (NSE, `.NS`) plus a gold ETF
2. **Validates** data quality — drops tickers without usable history over the window, then enforces no missing values / no negative prices / monotonic dates on the rest
3. **Estimates** time-varying risk using Exponentially Weighted Moving Average (EWMA) volatility and correlation, with correlation shrinkage
4. **Allocates** capital using Hierarchical Risk Parity (HRP) — clustering the ~49-name universe by correlation and splitting risk across clusters, no matrix inversion required
5. **Scales** total exposure toward a constant volatility target (capped leverage), holding the balance in cash at the risk-free rate
6. **Backtests** with walk-forward validation, expanding training windows, 1-day execution lag, EMA-smoothed targets, banded rebalancing, and transaction costs
7. **Benchmarks** against equal-weight and inverse-variance strategies
8. **Exports** metrics (Sharpe, max drawdown, turnover, cost drag) and an interactive Streamlit dashboard

## Why It Matters

Human investors guess allocations based on intuition, leading to concentration risk, panic selling, and no historical validation. Syntropy proves whether a strategy works *before* deploying capital, and the no-lookahead test guarantees the backtest isn't cheating by using future data.

## Key Features

- **From-scratch HRP** — no external portfolio libraries; implements distance matrix, linkage, quasi-diagonalization, recursive bisection, and constraint projection
- **Point-in-time correctness** — every feature uses only data available at its decision timestamp
- **Reproducible** — same config + same data = identical output
- **Volatility targeting** — post-allocation overlay scales exposure to a constant risk budget with capped leverage
- **Fast** — full pipeline on ~49 assets × 10 years runs in well under a minute
- **Portable** — single virtual environment, 8 core dependencies, runs on any machine

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| Data manipulation | Pandas, NumPy |
| Risk / clustering | SciPy |
| Validation | Pydantic |
| Storage | PyArrow / Parquet |
| Data source | yfinance |
| Visualization | Plotly, Streamlit |
| Testing | pytest, pytest-cov |

## Installation

```powershell
# Create virtual environment
python -m venv venv

# Activate
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Run the full pipeline

```powershell
.\venv\Scripts\python.exe scripts/run_pipeline.py
```

This downloads data, runs the walk-forward backtest, computes benchmarks, calculates metrics, and saves artifacts to `data/outputs/`.

### Run tests

```powershell
# All tests
.\venv\Scripts\python.exe -m pytest tests/ -v

# With coverage
.\venv\Scripts\python.exe -m pytest tests/ --cov=src --cov-report=term-missing
```

### Launch the dashboard

```powershell
streamlit run src/dashboard.py
```

Opens an interactive web UI at `http://localhost:8501` with:
- Cumulative return curves
- Drawdown profiles
- Weight evolution charts
- Performance metrics table

## Project Structure

```
config.py                     # Frozen dataclass configuration
src/
  domain.py                   # Pydantic models (Weights)
  features.py                 # Log returns, EWMA vol, EWMC correlation
  covariance.py               # Covariance assembly + PSD repair
  hrp.py                      # Hierarchical Risk Parity allocator
  benchmarks.py               # Equal-weight / inverse-variance strategies
  core/
    backtest.py               # Walk-forward simulation engine
  metrics.py                  # Sharpe, drawdown, turnover, cost drag
  reporting.py                # Parquet + JSON artifact saver
  ingestion.py                # Yahoo Finance download + validation
  dashboard.py                # Streamlit + Plotly UI
tests/
  conftest.py                 # Shared fixtures
  test_features.py
  test_covariance.py
  test_domain.py
  test_hrp.py
  test_backtest.py
  test_benchmarks.py
  test_lookahead.py           # Critical no-lookahead test
scripts/
  run_pipeline.py             # One-command entry point
data/
  processed/prices.parquet    # Cleaned price data
  outputs/
    backtest_results.parquet
    weights_history.parquet
    metrics.json
```

## How It Works

### 1. Ingestion
Downloads split/dividend-adjusted close prices via `yfinance` for the candidate universe in `config.py`. Tickers without usable history over the window (late listings, delisted symbols, gappy data) are dropped with a logged reason; the survivors are checked for missing/negative values and saved as immutable Parquet.

### 2. Feature Engineering
Computes point-in-time features:
- **Log returns**: `ln(P_t / P_{t-1})`
- **EWMA volatility**: annualized, span-configurable
- **EWMC correlation**: exponentially weighted moving correlation matrix

### 3. Covariance Assembly
Shrinks the correlation matrix toward a constant-correlation target (`corr_shrinkage`), then combines volatility and correlation into `Σ = D^{1/2} R D^{1/2}`. If the matrix is not positive semidefinite, eigenvalues are clipped and the matrix is reconstructed.

### 4. HRP Allocation
1. Distance matrix: `d(i,j) = sqrt(0.5 * (1 - ρ(i,j)))`
2. Single-linkage clustering via SciPy
3. Quasi-diagonalization to order similar assets adjacently
4. Recursive bisection: split sorted list in half, size each side inversely to its inverse-variance-weighted cluster variance, recurse
5. Constraint projection: clip weights to `[min_weight, max_weight]`, redistribute the residual across names with headroom, normalize to sum 1.0

### 5. Volatility-Target Overlay
Portfolio volatility at the HRP weights is read off the annualized covariance
(`sqrt(wᵀ Σ w)`). Exposure is scaled by `vol_target_annual / portfolio_vol`,
capped at `max_leverage`; the remaining fraction sits in cash (or is borrowed, if
levered) at the risk-free rate.

### 6. Walk-Forward Backtest
- Expanding training window from start date
- Test window advances by configurable step
- Weights decided at time `t` are applied to returns at `t+1` (execution lag)
- **Target smoothing**: the desired allocation is an EMA toward the raw HRP
  target (`weight_smoothing`); the leverage is left unsmoothed so it can
  de-risk quickly when volatility spikes
- **Banded rebalancing**: the smoothed book is only traded when it has drifted
  by more than `rebalance_band` (sum of absolute weight changes)
- Transaction costs deducted on rebalance dates

### 7. Metrics
- **Sharpe ratio**: `(mean excess return / std) * sqrt(252)`
- **Max drawdown**: worst peak-to-trough decline
- **Turnover**: average absolute weight change across rebalances
- **Cost drag**: return lost to transaction costs

## Example Output

```
                       Sharpe   Ann.Return   Ann.Vol   Max DD
Strategy (HRP+VT+EMA)    0.79      15.11%       11.53%   -13.62%
Equal weight            0.56      15.51%       17.03%   -38.00%
Inverse variance        0.58      14.99%       15.59%   -35.49%

Strategy turnover: 0.25%    Strategy cost drag: 0.55%
```

*Backtest period: 2015-01-01 to 2025-01-01 (test window runs 2017-2024 after the
2-year training warm-up). Universe: NIFTY 50 constituents (NSE, `.NS`) plus
`GOLDBEES.NS`; ~49 traded after auto-dropping names without full history. Prices
in INR — returns are ratios, so currency is irrelevant. `risk_free_rate_annual`
is 6% (Indian T-bill proxy). On this broader universe the strategy beats **both**
benchmarks on Sharpe, at roughly equal return but ~5-6 points less volatility and
less than half the max drawdown. Turnover is near-zero because the wide
`rebalance_band` (0.10) rarely fires on a 49-name book — tighten it for a more
responsive strategy.*

> **Survivorship bias caveat:** the ticker list is the *current* NIFTY 50
> membership applied back to 2015. Names that were in the index then but dropped
> out (and later poor performers) are absent, which flatters the backtest. A
> point-in-time constituent history would fix this; it is out of scope here.

## No-Lookahead Guarantee

`tests/test_lookahead.py` proves the strategy's outputs up to any cutoff date are identical, regardless of whether future spike data exists in the input. This guarantees zero future data leakage.

## Reproducibility

`test_reproducibility` runs the full backtest twice with identical config and data, asserting the return series, weight history, and metrics are bit-identical.

## Documentation

- [Architecture](ARCHITECTURE.md) — component diagram, data flow, design decisions
- [Testing](TESTING.md) — how to run tests, interpreting metrics
- [Decisions](DECISIONS.md) — architectural decision log
