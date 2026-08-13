# Syntropy

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Tests](https://img.shields.io/badge/tests-15%20passing-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

**A point-in-time quantitative pipeline that estimates time-varying risk via EWMA, allocates capital through a from-scratch Hierarchical Risk Parity (HRP) engine, and validates performance via walk-forward backtesting with provable zero future-data leakage. Outputs an interactive Streamlit dashboard for real-time strategy benchmarking.**

## What It Does

Syntropy replaces guesswork with a reproducible, mathematical portfolio allocation system:

1. **Downloads** adjusted daily prices from Yahoo Finance
2. **Validates** data quality (no missing values, no negative prices, monotonic dates)
3. **Estimates** time-varying risk using Exponentially Weighted Moving Average (EWMA) volatility and correlation
4. **Allocates** capital using Hierarchical Risk Parity (HRP) — clustering assets by correlation and splitting risk equally across clusters, no matrix inversion required
5. **Backtests** with walk-forward validation, expanding training windows, 1-day execution lag, and transaction costs
6. **Benchmarks** against equal-weight and inverse-variance strategies
7. **Exports** metrics (Sharpe, max drawdown, turnover, cost drag) and an interactive Streamlit dashboard

## Why It Matters

Human investors guess allocations based on intuition, leading to concentration risk, panic selling, and no historical validation. Syntropy proves whether a strategy works *before* deploying capital, and the no-lookahead test guarantees the backtest isn't cheating by using future data.

## Key Features

- **From-scratch HRP** — no external portfolio libraries; implements distance matrix, linkage, quasi-diagonalization, recursive bisection, and constraint projection
- **Point-in-time correctness** — every feature uses only data available at its decision timestamp
- **Reproducible** — same config + same data = identical output
- **Fast** — full pipeline on 5 assets × 5 years runs in under 30 seconds
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
Downloads adjusted close prices via `yfinance`, validates no missing/negative values, and saves immutable Parquet.

### 2. Feature Engineering
Computes point-in-time features:
- **Log returns**: `ln(P_t / P_{t-1})`
- **EWMA volatility**: annualized, span-configurable
- **EWMC correlation**: exponentially weighted moving correlation matrix

### 3. Covariance Assembly
Combines volatility and correlation into `Σ = D^{1/2} R D^{1/2}`. If the matrix is not positive semidefinite, eigenvalues are clipped and the matrix is reconstructed.

### 4. HRP Allocation
1. Distance matrix: `d(i,j) = sqrt(0.5 * (1 - ρ(i,j)))`
2. Single-linkage clustering via SciPy
3. Quasi-diagonalization to order similar assets adjacently
4. Recursive bisection: split sorted list in half, allocate capital inversely proportional to cluster variance, recurse
5. Constraint projection: clip weights to `[min_weight, max_weight]`, redistribute excess, normalize to sum 1.0

### 5. Walk-Forward Backtest
- Expanding training window from start date
- Test window advances by configurable step
- Weights decided at time `t` are applied to returns at `t+1` (execution lag)
- Transaction costs deducted on rebalance dates

### 6. Metrics
- **Sharpe ratio**: `(mean excess return / std) * sqrt(252)`
- **Max drawdown**: worst peak-to-trough decline
- **Turnover**: average absolute weight change across rebalances
- **Cost drag**: return lost to transaction costs

## Example Output

```
Strategy Sharpe: 0.672
Strategy Max DD: -27.67%
Strategy Ann. Return: 19.27%
Strategy Ann. Volatility: 22.71%
Strategy Turnover: 6.35%
Strategy Cost Drag: 0.38%

Equal Weight Sharpe: 0.833
Inverse Variance Sharpe: 0.699
```

*Backtest period: 2020-01-01 to 2025-01-01, symbols: AAPL, MSFT, NVDA, JPM, XOM.*

## No-Lookahead Guarantee

`tests/test_lookahead.py` proves the strategy's outputs up to any cutoff date are identical, regardless of whether future spike data exists in the input. This guarantees zero future data leakage.

## Reproducibility

`test_reproducibility` runs the full backtest twice with identical config and data, asserting the return series, weight history, and metrics are bit-identical.

## Documentation

- [Architecture](ARCHITECTURE.md) — component diagram, data flow, design decisions
- [Testing](TESTING.md) — how to run tests, interpreting metrics
- [Decisions](DECISIONS.md) — architectural decision log

## License

MIT
