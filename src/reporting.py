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
