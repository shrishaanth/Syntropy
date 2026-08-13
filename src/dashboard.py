import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
from pathlib import Path

st.set_page_config(page_title="Quantitative Portfolio Dashboard", layout="wide")

RESULTS_PATH = Path("data/outputs/backtest_results.parquet")
WEIGHTS_PATH = Path("data/outputs/weights_history.parquet")
METRICS_PATH = Path("data/outputs/metrics.json")


@st.cache_data
def load_data():
    if not RESULTS_PATH.exists() or not METRICS_PATH.exists():
        return None, None, None
    results = pd.read_parquet(RESULTS_PATH)
    weights = pd.read_parquet(WEIGHTS_PATH) if WEIGHTS_PATH.exists() else None
    with open(METRICS_PATH) as f:
        metrics_data = json.load(f)
    metrics = pd.Series(pd.json_normalize(metrics_data).iloc[0].to_dict())
    return results, weights, metrics


def cumulative_returns_figure(results: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for col in results.columns:
        cum = (1 + results[col]).cumprod()
        fig.add_trace(go.Scatter(x=cum.index, y=cum.values, mode="lines", name=col))
    fig.update_layout(title="Cumulative Returns", xaxis_title="Date", yaxis_title="Growth of $1", hovermode="x unified")
    return fig


def drawdown_figure(results: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for col in results.columns:
        cum = (1 + results[col]).cumprod()
        peak = cum.cummax()
        dd = (cum - peak) / peak
        fig.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines", name=col, fill="tozeroy"))
    fig.update_layout(title="Drawdown", xaxis_title="Date", yaxis_title="Drawdown", hovermode="x unified")
    return fig


def weight_evolution_figure(weights: pd.DataFrame) -> go.Figure:
    if weights is None or weights.empty:
        return go.Figure()
    fig = go.Figure()
    for col in weights.columns:
        fig.add_trace(go.Scatter(x=weights.index, y=weights[col], mode="lines", stackgroup="one", name=col))
    fig.update_layout(title="Weight Evolution", xaxis_title="Date", yaxis_title="Weight", hovermode="x unified")
    return fig


def main():
    st.title("Quantitative Portfolio Dashboard")
    st.sidebar.header("Configuration")

    results, weights, metrics = load_data()

    if results is None:
        st.warning("Artifacts not found. Run `python scripts/run_pipeline.py` first.")
        return

    if metrics is not None:
        st.sidebar.subheader("Metrics")
        for key in ["strategy.sharpe", "strategy.max_drawdown", "strategy.annualized_return"]:
            val = metrics.get(key, np.nan)
            if "drawdown" in key:
                st.sidebar.metric(key.replace("strategy.", "").replace("_", " ").title(), f"{val:.2%}")
            else:
                st.sidebar.metric(key.replace("strategy.", "").replace("_", " ").title(), f"{val:.3f}")

    tab1, tab2, tab3 = st.tabs(["Performance", "Drawdown", "Weights"])

    with tab1:
        st.plotly_chart(cumulative_returns_figure(results), use_container_width=True)

    with tab2:
        st.plotly_chart(drawdown_figure(results), use_container_width=True)

    with tab3:
        st.plotly_chart(weight_evolution_figure(weights), use_container_width=True)

    st.subheader("Metrics Table")
    if metrics is not None:
        flat = metrics.to_dict()
        rows = []
        for k, v in flat.items():
            rows.append({"Metric": k, "Value": v})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


if __name__ == "__main__":
    main()
