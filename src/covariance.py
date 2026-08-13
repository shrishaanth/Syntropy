import numpy as np
import pandas as pd


def build_covariance(vol: pd.Series, corr: pd.DataFrame) -> pd.DataFrame:
    """Assemble covariance matrix from volatility and correlation."""
    cov = np.outer(vol, vol) * corr.values
    return pd.DataFrame(cov, index=vol.index, columns=vol.index)


def psd_repair(cov: pd.DataFrame, epsilon: float = 1e-8) -> pd.DataFrame:
    """Repair covariance matrix to be positive semidefinite."""
    vals, vecs = np.linalg.eigh(cov.values)
    if (vals >= -epsilon).all():
        return cov
    vals = np.clip(vals, epsilon, None)
    repaired = vecs @ np.diag(vals) @ vecs.T
    return pd.DataFrame(repaired, index=cov.index, columns=cov.index)
