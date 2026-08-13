# Architecture

## Components

```
Config → Ingestion → Features → Covariance → HRP → Backtest → Metrics → Reporting → Dashboard
```

| Component | Responsibility |
|-----------|----------------|
| `config.py` | Immutable parameter container |
| `ingestion.py` | Download + validate market data from Yahoo Finance |
| `features.py` | Point-in-time returns, EWMA volatility, EWMC correlation |
| `covariance.py` | Assemble + repair valid covariance matrices |
| `hrp.py` | Cluster-based risk-parity allocation with constraints |
| `core/backtest.py` | Walk-forward simulation with execution lag and costs |
| `benchmarks.py` | Equal-weight and inverse-variance comparison strategies |
| `metrics.py` | Sharpe, drawdown, turnover, cost drag |
| `reporting.py` | Persist Parquet + JSON artifacts |
| `dashboard.py` | Interactive Streamlit/Plotly visualization |

## Data Flow

1. **Ingestion**: `yfinance.download()` → validated `prices.parquet`
2. **Features**: prices → log returns → EWMA vol → EWMC correlation
3. **Covariance**: assemble `Σ = D R D`, repair PSD if needed
4. **HRP**: distance matrix → linkage → quasi-diagonalize → recursive bisection → constraint clip → weights sum to 1
5. **Backtest**: expanding walk-forward windows, fit on train, allocate on cov, apply to test with 1-day lag, deduct costs
6. **Metrics**: Sharpe, max drawdown, turnover, cost drag for strategy + benchmarks
7. **Reporting**: save `backtest_results.parquet`, `weights_history.parquet`, `metrics.json`
8. **Dashboard**: read artifacts → Plotly charts → web UI at localhost:8501

## Why HRP?

Traditional mean-variance optimization inverts the covariance matrix, which explodes when assets are highly correlated. HRP uses clustering and recursive bisection — no inversion, robust to noise, and interpretable (produces clusters).

## Why Point-in-Time Matters

A backtest must not cheat by using future data. The `test_lookahead.py` suite proves that the strategy's outputs up to any cutoff date are identical, regardless of whether future spike data exists in the input.

## Interfaces

- **Features → Covariance**: Volatility vector (`pd.Series`) + correlation matrix (`pd.DataFrame`) → valid covariance (`pd.DataFrame`)
- **Covariance → HRP**: Valid covariance DataFrame → weight dict summing to ~1.0
- **HRP → Backtest**: Weight dict → `pd.Series` for dot-product with returns
- **Backtest → Metrics**: Daily return `pd.Series` → metrics dict
- **Reporting → Dashboard**: Parquet + JSON files on disk → Plotly charts
