# Syntropy

**A reproducible, systematic quantitative portfolio platform using Hierarchical Risk Parity (HRP) and walk-forward backtesting.**

## Setup

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```powershell
# Full pipeline
.\venv\Scripts\python.exe scripts/run_pipeline.py

# Tests
.\venv\Scripts\python.exe -m pytest tests/ -v

# Dashboard
streamlit run src/dashboard.py
```

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
tests/                        # pytest suite
scripts/
  run_pipeline.py             # One-command entry point
data/
  processed/prices.parquet    # Cleaned price data
  outputs/                    # Backtest results, weights, metrics
```

## Key Concepts

- **HRP**: Clusters assets by correlation, allocates risk equally across clusters — no matrix inversion, robust to noise.
- **Point-in-time correctness**: Every feature uses only data available at its decision timestamp. The no-lookahead test proves this.
- **Reproducibility**: Same config + same data = bit-identical output. Fixed seeds where applicable.
