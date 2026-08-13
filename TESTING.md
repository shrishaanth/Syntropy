# Testing

## Running Tests

```powershell
# All tests
.\venv\Scripts\python.exe -m pytest tests/ -v

# With coverage
.\venv\Scripts\python.exe -m pytest tests/ --cov=src --cov-report=term-missing

# Specific test file
.\venv\Scripts\python.exe -m pytest tests/test_lookahead.py -v
```

## Test Files

| File | What it tests |
|------|---------------|
| `test_features.py` | Log returns correctness, EWMA vol shape, EWMC correlation bounds |
| `test_covariance.py` | Covariance assembly math, PSD repair validity |
| `test_domain.py` | Weights Pydantic model validation |
| `test_hrp.py` | HRP weights sum to 1, respect min/max constraints |
| `test_backtest.py` | Walk-forward produces no NaN, weights sum to 1, window count correct |
| `test_benchmarks.py` | Equal-weight and inverse-variance allocations |
| `test_lookahead.py` | **Critical**: no future data leakage, reproducibility |

## No-Lookahead Test

`test_no_lookahead_guarantee` creates synthetic price data with a spike injected at a future date. It runs the backtest up to a cutoff date twice — once with the spike, once without — and asserts the strategy returns are identical to `1e-10` tolerance. This proves zero future data leakage.

## Interpreting Metrics

| Metric | Description | Good Range |
|--------|-------------|------------|
| Sharpe | Risk-adjusted return (annualized) | 0.5 – 2.0 |
| Max Drawdown | Worst peak-to-trough loss | 0% – -50% |
| Annualized Return | Compound annual growth | Positive vs benchmarks |
| Annualized Volatility | Annualized std dev of returns | 10% – 30% |
| Turnover | Average absolute weight change | Lower is better |
| Cost Drag | Return lost to transaction costs | Near 0% |

## Reproducibility

`test_reproducibility` runs the full backtest twice with the same config and data, then asserts the return series, weight history, and JSON metrics are identical. If this fails, something is non-deterministic.
