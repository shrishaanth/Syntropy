#!/usr/bin/env python3
"""One-command pipeline entry point."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config
from src.ingestion import download_and_save
from src.core.backtest import Strategy
from src.benchmarks import equal_weight, inverse_variance
from src.metrics import calculate_metrics
from src.reporting import save_artifacts
import pandas as pd
import numpy as np


def main():
    config = Config()

    print("=" * 60)
    print("QUANTITATIVE PORTFOLIO PIPELINE")
    print("=" * 60)

    print("\n[1/4] Downloading and validating data...")
    prices_path = download_and_save(config)
    print(f"Saved prices to {prices_path}")

    print("\n[2/4] Running walk-forward backtest...")
    prices = pd.read_parquet(prices_path)
    strategy = Strategy(
        train_window=config.train_years * 252,
        test_window=config.test_months * 21,
        transaction_cost=config.transaction_cost_bps / 10000,
    )
    results, weights_df = strategy.run(prices)
    print(f"Strategy returns: {len(results)} days, {len(weights_df)} rebalances")

    print("\n[3/4] Computing benchmarks...")
    asset_prices = prices.drop(columns=config.benchmarks, errors="ignore") if hasattr(config, 'benchmarks') else prices
    log_returns = pd.DataFrame(np.log(asset_prices / asset_prices.shift(1))).dropna()

    benchmark_returns = {}
    ew_weights = equal_weight(log_returns, config)
    benchmark_returns["equal_weight"] = log_returns.dot(pd.Series(ew_weights))

    iv_weights = inverse_variance(log_returns, config)
    benchmark_returns["inverse_variance"] = log_returns.dot(pd.Series(iv_weights))

    print("\n[4/4] Calculating metrics and saving artifacts...")
    strategy_returns = results["Strategy"].dropna() if "Strategy" in results.columns else results.iloc[:, 0].dropna()

    metrics = calculate_metrics(
        strategy_returns,
        {k: v.dropna() for k, v in benchmark_returns.items()},
        weights_df,
        config,
    )

    paths = save_artifacts(results, weights_df, metrics, config)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Results:   {paths['results']}")
    print(f"Weights:   {paths['weights']}")
    print(f"Metrics:   {paths['metrics']}")
    print(f"\nStrategy Sharpe: {metrics['strategy']['sharpe']:.3f}")
    print(f"Strategy Max DD: {metrics['strategy']['max_drawdown']:.2%}")


if __name__ == "__main__":
    main()
