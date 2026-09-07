import numpy as np
import pandas as pd


def shrink_correlation(corr: pd.DataFrame, delta: float) -> pd.DataFrame:
    if delta <= 0.0:
        return corr
    delta = min(delta, 1.0)

    n = corr.shape[0]
    off_diag = corr.values[~np.eye(n, dtype=bool)]
    mean_corr = off_diag.mean() if off_diag.size else 0.0

    target = np.full((n, n), mean_corr)
    np.fill_diagonal(target, 1.0)

    shrunk = (1.0 - delta) * corr.values + delta * target
    np.fill_diagonal(shrunk, 1.0)
    return pd.DataFrame(shrunk, index=corr.index, columns=corr.columns)


def build_covariance(vol: pd.Series, corr: pd.DataFrame) -> pd.DataFrame:
    cov = np.outer(vol, vol) * corr.values
    return pd.DataFrame(cov, index=vol.index, columns=vol.index)


def psd_repair(cov: pd.DataFrame, epsilon: float = 1e-8) -> pd.DataFrame:
    vals, vecs = np.linalg.eigh(cov.values)
    if (vals >= -epsilon).all():
        return cov
    vals = np.clip(vals, epsilon, None)
    repaired = vecs @ np.diag(vals) @ vecs.T
    return pd.DataFrame(repaired, index=cov.index, columns=cov.index)
