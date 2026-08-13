# Decisions

## Why No GARCH in v1

GARCH(1,1) is more statistically appealing than EWMA, but it requires convergence checks, per-asset loops, and fallback logic. EWMA is transparent, single-pass, and sufficient for a v1 demo.

## Why Pure pandas Over vectorbt / Backtrader

The backtest engine is pure pandas to keep execution lag visible and controllable. Backtesting libraries hide the lag logic, which defeats the educational purpose of this project.

## Why HRP Over Markowitz

Mean-variance optimization requires matrix inversion and fails when covariance is near-singular. HRP is robust to noisy covariance estimates, interpretable (produces clusters), and does not require inversion.

## Why Expanding Window Over Rolling

Expanding window is the most realistic for a growing dataset. Rolling window discards older data that may still be relevant.

## Why Parquet Over CSV

Parquet is columnar, compressed, schema-preserving, and fast to read/write. CSV is slow, large, and loses type information.

## Why No Docker

Local execution target; containerization adds no value for a single-command research tool that reads no external services beyond Yahoo Finance.

## Why No scikit-learn

Not needed for HRP or EWMA. SciPy provides linkage and distance matrix functionality without the overhead.

## Why Frozen Config

Eliminates scattered magic numbers, prevents accidental mutation mid-run, and makes the pipeline reproducible — the config object can be serialized into the metrics log.
