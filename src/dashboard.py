import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
from pathlib import Path
from dataclasses import replace

# `streamlit run src/dashboard.py` puts src/ on sys.path, not the project
# root — same fix scripts/run_pipeline.py already needed for `config`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config
from src.core.backtest import Strategy
from src.benchmarks import benchmark_returns
from src.metrics import calculate_metrics
from src.features import ewma_vol, ewmc_corr
from src.covariance import build_covariance, psd_repair, shrink_correlation
from src.hrp import allocate
from src.ingestion import download_and_save

st.set_page_config(page_title="Quantitative Portfolio Dashboard", layout="wide")

RESULTS_PATH = Path("data/outputs/backtest_results.parquet")
WEIGHTS_PATH = Path("data/outputs/weights_history.parquet")
METRICS_PATH = Path("data/outputs/metrics.json")
HISTORY_PATH = Path("data/outputs/history/metrics_history.csv")
PRICES_PATH = Path("data/processed/prices.parquet")


@st.cache_data
def load_scheduled_outputs():
    """The last committed pipeline run — refreshed automatically on weekday
    afternoons by .github/workflows/refresh_pipeline.yml, or manually via
    `python scripts/run_pipeline.py`."""
    if not RESULTS_PATH.exists() or not METRICS_PATH.exists():
        return None, None, None
    results = pd.read_parquet(RESULTS_PATH)
    weights = pd.read_parquet(WEIGHTS_PATH) if WEIGHTS_PATH.exists() else None
    with open(METRICS_PATH) as f:
        metrics_data = json.load(f)
    metrics = pd.Series(pd.json_normalize(metrics_data).iloc[0].to_dict())
    return results, weights, metrics


@st.cache_data
def load_history():
    if not HISTORY_PATH.exists():
        return None
    return pd.read_csv(HISTORY_PATH, parse_dates=["as_of"])


def load_prices():
    if not PRICES_PATH.exists():
        return None
    return pd.read_parquet(PRICES_PATH)


def cumulative_returns_figure(results: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for col in results.columns:
        series = results[col].dropna()
        cum = np.exp(series.cumsum())
        fig.add_trace(go.Scatter(x=cum.index, y=cum.values, mode="lines", name=col))
    fig.update_layout(title="Cumulative Returns", xaxis_title="Date", yaxis_title="Growth of $1", hovermode="x unified")
    return fig


def drawdown_figure(results: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for col in results.columns:
        series = results[col].dropna()
        cum = np.exp(series.cumsum())
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


def history_figure(history: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history["as_of"], y=history["strategy_sharpe"], mode="lines+markers", name="Strategy"))
    if "equal_weight_sharpe" in history.columns:
        fig.add_trace(go.Scatter(x=history["as_of"], y=history["equal_weight_sharpe"], mode="lines+markers", name="Equal weight"))
    if "inverse_variance_sharpe" in history.columns:
        fig.add_trace(go.Scatter(x=history["as_of"], y=history["inverse_variance_sharpe"], mode="lines+markers", name="Inverse variance"))
    fig.update_layout(title="Sharpe Across Scheduled Reruns", xaxis_title="As-of date", yaxis_title="Sharpe", hovermode="x unified")
    return fig


@st.cache_data(show_spinner=False)
def run_custom_backtest(prices: pd.DataFrame, base_config: Config, overrides: dict):
    """Re-runs the walk-forward backtest in-process with a few knobs
    overridden, entirely in memory — nothing here touches data/outputs/,
    so it can't clobber the scheduled run's committed artifacts. Cached so
    revisiting an already-tried parameter combination is instant; a novel
    one still pays the full walk-forward cost (each rebalance re-fits an
    EWM correlation matrix across the whole universe, which is the real
    bottleneck — see the window picker in the caller for why it matters)."""
    cfg = replace(base_config, **overrides)
    strategy = Strategy(
        train_window=cfg.train_years * 252,
        test_window=cfg.test_months * 21,
        transaction_cost=cfg.transaction_cost_bps / 10000,
    )
    strategy.config = cfg  # same override pattern the test suite uses
    results, weights_df, costs = strategy.run(prices)

    log_returns = np.log(prices / prices.shift(1)).dropna()
    bench_returns = benchmark_returns(log_returns, cfg)

    strategy_returns = results["Strategy"].dropna()
    metrics = calculate_metrics(
        strategy_returns,
        {k: v.dropna() for k, v in bench_returns.items()},
        weights_df,
        cfg,
        costs=costs,
    )
    results_full = results.copy()
    for name, series in bench_returns.items():
        results_full[name] = series.reindex(results_full.index)
    return results_full, weights_df, metrics


def current_target_weights(prices: pd.DataFrame, cfg: Config):
    """What the allocator would hold right now: the same training step the
    backtest runs at each rebalance, applied once to the most recent
    train_years window — no backtest, just today's target."""
    log_returns = np.log(prices / prices.shift(1)).dropna()
    window = cfg.train_years * 252
    train_data = log_returns.iloc[-window:]

    vol = ewma_vol(train_data, span=cfg.ewma_span)
    corr = ewmc_corr(train_data, span=cfg.corr_span)
    corr = shrink_correlation(corr, cfg.corr_shrinkage)
    cov = build_covariance(vol, corr)
    cov = psd_repair(cov)

    weights = pd.Series(allocate(cov, corr, cfg), index=prices.columns).sort_values(ascending=False)

    port_vol = float(np.sqrt(weights.values @ cov.values @ weights.values))
    leverage = min(cfg.vol_target_annual / port_vol, cfg.max_leverage) if port_vol > 0 else 1.0
    return weights, leverage, port_vol


def main():
    st.title("Quantitative Portfolio Dashboard")
    base_config = Config()

    scheduled_results, scheduled_weights, scheduled_metrics = load_scheduled_outputs()
    history = load_history()

    if scheduled_results is None:
        st.warning("No pipeline run found yet. Run `python scripts/run_pipeline.py` once to generate `data/outputs/`.")
        return

    if scheduled_metrics is not None:
        st.sidebar.subheader("Latest Scheduled Run")
        for key in ["strategy.sharpe", "strategy.max_drawdown", "strategy.annualized_return"]:
            val = scheduled_metrics.get(key, np.nan)
            label = key.replace("strategy.", "").replace("_", " ").title()
            fmt = f"{val:.2%}" if "drawdown" in key or "return" in key else f"{val:.3f}"
            st.sidebar.metric(label, fmt)
        as_of = scheduled_metrics.get("config.end_date", "unknown")
        st.sidebar.caption(f"Data as of {as_of} — refreshed by GitHub Actions on weekday afternoons.")

    tabs = st.tabs(["Performance", "Drawdown", "Weights", "History", "Current Targets", "Try Your Own"])

    with tabs[0]:
        st.plotly_chart(cumulative_returns_figure(scheduled_results), use_container_width=True)

    with tabs[1]:
        st.plotly_chart(drawdown_figure(scheduled_results), use_container_width=True)

    with tabs[2]:
        st.plotly_chart(weight_evolution_figure(scheduled_weights), use_container_width=True)

    with tabs[3]:
        if history is not None and len(history) > 1:
            st.plotly_chart(history_figure(history), use_container_width=True)
            st.dataframe(history, use_container_width=True)
        else:
            st.info(
                "One row is appended here per pipeline run. Since the scheduled "
                "job runs once a day, this fills in over the next few days."
            )

    with tabs[4]:
        st.subheader("What the allocator would hold today")
        col_refresh, _ = st.columns([1, 4])
        if col_refresh.button("🔄 Refresh price data"):
            with st.spinner("Downloading latest prices from Yahoo Finance..."):
                download_and_save(base_config)
            st.cache_data.clear()
            st.rerun()

        prices = load_prices()
        if prices is None:
            st.warning("No cached price data on this machine yet — click **Refresh price data** above, or run the pipeline once.")
        else:
            st.caption(f"Using local price cache through {prices.index.max().date()}.")
            weights, leverage, port_vol = current_target_weights(prices, base_config)
            col1, col2 = st.columns([2, 1])
            with col1:
                fig = go.Figure(go.Bar(x=weights.index, y=weights.values))
                fig.update_layout(title="Target Weights (unlevered)", yaxis_title="Weight")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.metric("Suggested leverage", f"{leverage:.2f}x")
                st.metric("Portfolio vol (unlevered)", f"{port_vol:.2%}")
                st.caption(
                    "Leverage scales exposure toward the config's volatility "
                    "target; the rest sits in cash at the risk-free rate."
                )

    with tabs[5]:
        st.subheader("Backtest with your own parameters")
        prices = load_prices()
        if prices is None:
            st.warning("No cached price data on this machine yet — visit **Current Targets** and click Refresh, or run the pipeline once.")
        else:
            # Each rebalance re-fits an EWM correlation matrix across the
            # whole ~49-asset universe, which dominates runtime (~0.5-0.7s
            # per rebalance on a typical machine) — the full history is
            # ~114 rebalances, so a full-history run can take a minute or
            # more. Default to a shorter recent window so tweaking a
            # parameter and re-running stays interactive; opt into more
            # history when you actually need it.
            window_choice = st.selectbox(
                "Backtest window",
                ["Last 1 year (fast, ~10s)", "Last 3 years (~25s)", "Last 5 years (~40s)", "Full history (~60-90s)"],
                index=0,
            )
            window_years = {"Last 1 year (fast, ~10s)": 1, "Last 3 years (~25s)": 3, "Last 5 years (~40s)": 5, "Full history (~60-90s)": None}[window_choice]

            c1, c2, c3, c4 = st.columns(4)
            vol_target = c1.slider("Vol target (annual)", 0.02, 0.30, base_config.vol_target_annual, 0.01)
            max_lev = c1.slider("Max leverage", 1.0, 3.0, base_config.max_leverage, 0.1)
            band = c2.slider("Rebalance band", 0.0, 0.30, base_config.rebalance_band, 0.01)
            smoothing = c2.slider("Weight smoothing (EMA)", 0.05, 1.0, base_config.weight_smoothing, 0.05)
            shrinkage = c3.slider("Correlation shrinkage", 0.0, 1.0, base_config.corr_shrinkage, 0.05)
            cost_bps = c3.slider("Transaction cost (bps)", 0.0, 50.0, base_config.transaction_cost_bps, 1.0)
            min_w = c4.slider("Min asset weight", 0.0, 0.10, base_config.min_asset_weight, 0.01)
            max_w = c4.slider("Max asset weight", 0.10, 0.50, base_config.max_asset_weight, 0.01)

            if st.button("▶ Run backtest", type="primary"):
                overrides = dict(
                    vol_target_annual=vol_target,
                    max_leverage=max_lev,
                    rebalance_band=band,
                    weight_smoothing=smoothing,
                    corr_shrinkage=shrinkage,
                    transaction_cost_bps=cost_bps,
                    min_asset_weight=min_w,
                    max_asset_weight=max_w,
                )
                sliced_prices = prices if window_years is None else prices.tail((base_config.train_years + window_years) * 252)
                with st.spinner(f"Running walk-forward backtest over {window_choice.split(' (')[0].lower()}..."):
                    custom_results, custom_weights, custom_metrics = run_custom_backtest(sliced_prices, base_config, overrides)
                st.session_state["custom_results"] = custom_results
                st.session_state["custom_metrics"] = custom_metrics

            if "custom_results" in st.session_state:
                cm = st.session_state["custom_metrics"]
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Sharpe", f"{cm['strategy']['sharpe']:.3f}")
                m2.metric("Max Drawdown", f"{cm['strategy']['max_drawdown']:.2%}")
                m3.metric("Ann. Return", f"{cm['strategy']['annualized_return']:.2%}")
                m4.metric("Turnover", f"{cm['strategy']['turnover']:.2%}")
                st.plotly_chart(
                    cumulative_returns_figure(st.session_state["custom_results"]),
                    use_container_width=True,
                )
            else:
                st.info("Adjust parameters above, then click Run backtest. Nothing here is saved to disk.")

    st.subheader("Latest Scheduled Run — Metrics Table")
    if scheduled_metrics is not None:
        # Value mixes floats, dates, and config.symbols (a list) — stringify
        # uniformly so pyarrow serializes cleanly instead of falling back
        # with a console warning on the mixed-type column.
        rows = [{"Metric": k, "Value": ", ".join(v) if isinstance(v, list) else str(v)} for k, v in scheduled_metrics.to_dict().items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


if __name__ == "__main__":
    main()
