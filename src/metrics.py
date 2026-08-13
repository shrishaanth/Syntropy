import pandas as pd
import numpy as np


def sharpe_ratio(returns: pd.Series, risk_free_rate_annual: float) -> float:
    """Compute annualized Sharpe ratio."""
    rf_daily = risk_free_rate_annual / 252
    excess = returns - rf_daily
    std = excess.std()
    if std == 0 or np.isnan(std):
        return np.nan
    return (excess.mean() / std) * np.sqrt(252)


def max_drawdown(returns: pd.Series) -> float:
    """Compute max drawdown from returns."""
    cumulative = np.exp(returns.cumsum())
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()


def annualized_return(returns: pd.Series) -> float:
    """Compute annualized return."""
    return returns.mean() * 252


def annualized_volatility(returns: pd.Series) -> float:
    """Compute annualized volatility."""
    return returns.std() * np.sqrt(252)


def turnover(weights_df: pd.DataFrame) -> float:
    """Compute average turnover across rebalances."""
    if len(weights_df) < 2:
        return 0.0
    changes = weights_df.diff().abs().iloc[1:]
    return changes.sum(axis=1).mean()


def cost_drag(strategy_returns: pd.Series, gross_returns: pd.Series) -> float:
    """Compute cost drag as fraction of gross return."""
    if gross_returns.sum() == 0:
        return 0.0
    return (gross_returns.sum() - strategy_returns.sum()) / gross_returns.sum()


def calculate_metrics(
    strategy_returns: pd.Series,
    benchmark_returns: dict[str, pd.Series],
    weights_df: pd.DataFrame,
    config,
    costs: pd.Series = None,
) -> dict:
    """Calculate all performance metrics."""
    gross_returns = strategy_returns.copy()
    if costs is not None:
        costs_aligned = costs.reindex(gross_returns.index).fillna(0)
        gross_returns = gross_returns + costs_aligned

    metrics = {
        "strategy": {
            "sharpe": sharpe_ratio(strategy_returns, config.risk_free_rate_annual),
            "max_drawdown": max_drawdown(strategy_returns),
            "annualized_return": annualized_return(strategy_returns),
            "annualized_volatility": annualized_volatility(strategy_returns),
            "turnover": turnover(weights_df),
            "cost_drag": cost_drag(strategy_returns, gross_returns),
        },
        "config": {
            "symbols": list(config.symbols),
            "start_date": config.start_date.isoformat(),
            "end_date": config.end_date.isoformat(),
            "ewma_span": config.ewma_span,
            "corr_span": config.corr_span,
            "train_years": config.train_years,
            "test_months": config.test_months,
            "transaction_cost_bps": config.transaction_cost_bps,
        },
    }

    for name, returns in benchmark_returns.items():
        metrics[name] = {
            "sharpe": sharpe_ratio(returns, config.risk_free_rate_annual),
            "max_drawdown": max_drawdown(returns),
            "annualized_return": annualized_return(returns),
            "annualized_volatility": annualized_volatility(returns),
        }

    return metrics
