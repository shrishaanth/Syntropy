import json
from pathlib import Path

import pandas as pd


def save_artifacts(
    results: pd.DataFrame,
    weights_df: pd.DataFrame,
    metrics: dict,
    config,
) -> dict[str, Path]:
    output_dir = Path("data/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "backtest_results.parquet"
    weights_path = output_dir / "weights_history.parquet"
    metrics_path = output_dir / "metrics.json"

    results.to_parquet(results_path)
    weights_df.to_parquet(weights_path)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    return {
        "results": results_path,
        "weights": weights_path,
        "metrics": metrics_path,
    }


def append_history(metrics: dict, results: pd.DataFrame) -> Path:
    """Append this run's headline numbers to a running CSV, keyed by the
    as-of date of the data (the last date in the backtest results) rather
    than wall-clock time — a same-day rerun replaces its row instead of
    duplicating it. Lets the dashboard plot Sharpe/drawdown trends across
    scheduled reruns instead of only ever showing the latest snapshot.
    """
    history_dir = Path("data/outputs/history")
    history_dir.mkdir(parents=True, exist_ok=True)
    history_path = history_dir / "metrics_history.csv"

    as_of = results.index.max().date().isoformat()
    row = {
        "as_of": as_of,
        "strategy_sharpe": metrics["strategy"]["sharpe"],
        "strategy_max_drawdown": metrics["strategy"]["max_drawdown"],
        "strategy_annualized_return": metrics["strategy"]["annualized_return"],
        "strategy_annualized_volatility": metrics["strategy"]["annualized_volatility"],
        "strategy_turnover": metrics["strategy"]["turnover"],
        "strategy_cost_drag": metrics["strategy"]["cost_drag"],
    }
    for name in ("equal_weight", "inverse_variance"):
        if name in metrics:
            row[f"{name}_sharpe"] = metrics[name]["sharpe"]
            row[f"{name}_max_drawdown"] = metrics[name]["max_drawdown"]

    if history_path.exists():
        history = pd.read_csv(history_path)
        history = history[history["as_of"] != as_of]
        history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)
    else:
        history = pd.DataFrame([row])

    history.sort_values("as_of").to_csv(history_path, index=False)
    return history_path
