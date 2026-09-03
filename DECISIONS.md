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

## Why a Diversified Cross-Asset Universe

HRP earns its keep when the correlation matrix has genuine block structure. Five
correlated US large-caps gave it almost nothing to cluster, so it degenerated
toward equal weight and then lost to it on estimation noise. The universe is now
~18 names spanning equity sectors plus treasuries, gold, and REITs, which is
where the clustering and recursive bisection actually change the allocation.

## Why a Volatility-Target Overlay

HRP equalises *risk contribution* but says nothing about the portfolio's total
risk level, which drifts with the market. A post-allocation overlay scales
exposure toward a constant `vol_target_annual` (capped at `max_leverage`), with
the remainder in cash at the risk-free rate. This makes drawdown and
cross-strategy Sharpe comparisons meaningful and is the standard risk-parity
construction. It is deliberately a separate, legible step rather than folded into
the allocator.

## Why Banded Rebalancing

Rebalancing to target every period pays transaction costs on noise. The engine
now only trades when the target book has drifted past `rebalance_band` (sum of
absolute weight changes); otherwise it holds the existing book. The band is
applied to the post-overlay book so both allocation drift and leverage drift
count toward the trade decision.

## Why EMA-Smooth the Allocation but Not the Leverage

With ~18 assets the EWMA covariance is noisy enough that rebalancing straight to
the raw HRP target churned the whole book every period (~20% turnover, ~4% cost
drag). Smoothing the *desired allocation* with an EMA (`weight_smoothing`) cut
that to ~4.5% turnover with no loss of return. The *leverage* is deliberately
left unsmoothed: a parameter sweep showed smoothing it makes the strategy slow
to de-risk into volatility spikes, which widened the max drawdown from ~17% to
~28%. Allocation should be sticky; risk scaling should be responsive.

## Why a 10-Year Sample

A single 2020-2025 window put the entire walk-forward test inside the 2022 rate
shock, which is the worst possible regime for a bond-inclusive risk-parity book.
Starting in 2015 spans the 2018 selloff, the COVID crash, 2022, and the
recoveries, so the reported metrics reflect several regimes rather than one.

## Why Frozen Config

Eliminates scattered magic numbers, prevents accidental mutation mid-run, and makes the pipeline reproducible — the config object can be serialized into the metrics log.
