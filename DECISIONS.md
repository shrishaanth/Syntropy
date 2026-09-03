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

## Why Inverse-Variance Cluster Variance in HRP

The recursive bisection step needs each half-cluster's variance to decide the
split. The canonical HRP (Lopez de Prado, 2016) evaluates that variance with the
cluster held at **inverse-variance weights**, not equal weights. Equal weighting
makes the split sensitive to how many names sit on each side rather than to how
risky each side actually is, which biases capital toward larger clusters. The
inverse-variance definition is a one-liner and restores the intended behaviour.

## Why Correlation Shrinkage

The EWMA correlation estimate is noisy on a small universe. Each rebalance shrinks
it toward a constant-correlation target (mean off-diagonal) by a fixed intensity
(`corr_shrinkage`, default 0.10). A fixed intensity keeps the step transparent and
reproducible, consistent with the "single-pass, no convergence checks" rationale
for EWMA over GARCH. On the 5-asset demo universe the effect is near-neutral; it
matters once the universe is widened enough to have real cluster structure.

## Why Frozen Config

Eliminates scattered magic numbers, prevents accidental mutation mid-run, and makes the pipeline reproducible — the config object can be serialized into the metrics log.
