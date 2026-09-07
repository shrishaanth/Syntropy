import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

from src.covariance import build_covariance, psd_repair


def _distance_matrix(corr: pd.DataFrame) -> pd.DataFrame:
    d = np.sqrt(np.clip(0.5 * (1 - corr.values), 0.0, None))
    np.fill_diagonal(d, 0.0)
    return pd.DataFrame(d, index=corr.index, columns=corr.index)


def _allocate(cov: pd.DataFrame, corr: pd.DataFrame, min_w: float, max_w: float) -> dict[str, float]:
    d = _distance_matrix(corr)
    condensed = squareform(d.values, checks=False)
    Z = linkage(condensed, method="single")
    order = leaves_list(Z)
    items = [cov.index[i] for i in order]
    sub_cov = cov.values[np.ix_(order, order)]
    weights = _bisect(sub_cov, items)
    weights = _clip_constraints(weights, min_w, max_w)
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


def _bisect(cov: np.ndarray, items: list[str]) -> dict[str, float]:
    if len(items) == 1:
        return {items[0]: 1.0}
    split = len(items) // 2
    left_items = items[:split]
    right_items = items[split:]
    left_idx = list(range(split))
    right_idx = list(range(split, len(items)))
    var_left = _cluster_var(cov, left_idx)
    var_right = _cluster_var(cov, right_idx)
    alloc_left = 1 - var_left / (var_left + var_right)
    w_left = _bisect(cov[np.ix_(left_idx, left_idx)], left_items)
    w_right = _bisect(cov[np.ix_(right_idx, right_idx)], right_items)
    return {k: v * alloc_left for k, v in w_left.items()} | {
        k: v * (1 - alloc_left) for k, v in w_right.items()
    }


def _inverse_variance_weights(cov: np.ndarray) -> np.ndarray:
    ivp = 1.0 / np.diag(cov)
    return ivp / ivp.sum()


def _cluster_var(cov: np.ndarray, idx: list[int]) -> float:
    sub = cov[np.ix_(idx, idx)]
    w = _inverse_variance_weights(sub)
    return float(w @ sub @ w)


def _clip_constraints(weights: dict[str, float], min_w: float, max_w: float) -> dict[str, float]:
    w = {k: float(v) for k, v in weights.items()}
    for _ in range(50):
        w = {k: float(np.clip(v, min_w, max_w)) for k, v in w.items()}
        residual = 1.0 - sum(w.values())
        if abs(residual) < 1e-12:
            return w
        if residual > 0:
            headroom = {k: max_w - v for k, v in w.items() if v < max_w}
        else:
            headroom = {k: v - min_w for k, v in w.items() if v > min_w}
        total_headroom = sum(headroom.values())
        if total_headroom <= 1e-15:
            break
        for k, room in headroom.items():
            w[k] += residual * (room / total_headroom)
    w = {k: float(np.clip(v, min_w, max_w)) for k, v in w.items()}
    total = sum(w.values())
    return {k: v / total for k, v in w.items()}


def allocate(cov: pd.DataFrame, corr: pd.DataFrame, config) -> dict[str, float]:
    return _allocate(cov, corr, config.min_asset_weight, config.max_asset_weight)
