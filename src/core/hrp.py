import numpy as np
import pandas as pd
from typing import Tuple, List
import scipy.linalg as la
from .risk_estimators import EWMARiskEstimator
class HierarchicalRiskParity:

    def __init__(self, min_var: float = 1e-6):
        self.min_var = min_var

    def _quasidiagonalization(self, cov_matrix: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        cov = cov_matrix.values
        n = cov.shape[0]

        d = np.sqrt(np.diag(cov))
        D_inv = np.diag(1 / d)
        corr = D_inv @ cov @ D_inv

        A = corr.copy()
        W = np.eye(n)
        hierarchical_tree = []


        ones = np.ones(n)
        mvp_weights = np.linalg.solve(cov, ones)
        mvp_weights /= np.sum(mvp_weights)

        mvp_variance = np.dot(mvp_weights, cov @ mvp_weights)

        if mvp_variance > self.min_var:
            W = np.linalg.inv(np.sqrt(mvp_variance) * np.eye(n) + cov - mvp_variance * np.outer(mvp_weights, mvp_weights))
            return W, hierarchical_tree

        for i in range(n):
            indices = [j for j in range(n) if j != i]
            sub_cov = cov[np.ix_(indices, indices)]
            sub_ones = np.ones(len(indices))
            sub_mvp_weights = np.linalg.solve(sub_cov, sub_ones)
            sub_mvp_weights /= np.sum(sub_mvp_weights)

            sub_variance = np.dot(sub_mvp_weights, sub_cov @ sub_mvp_weights)

            if sub_variance > self.min_var:
                W = np.eye(n)
                W[indices, indices] = sub_mvp_weights
                return W, hierarchical_tree

        return np.eye(n), hierarchical_tree

    def _recursive_bisection(self, W: np.ndarray, hierarchical_tree: List) -> List:
        n = W.shape[0]

        if n == 1:
            return hierarchical_tree

        half = n // 2
        left = W[:half, :half]
        right = W[half:, half:]

        left_tree = self._recursive_bisection(left, hierarchical_tree)
        right_tree = self._recursive_bisection(right, hierarchical_tree)

        hierarchical_tree.append((left_tree, right_tree))

        return hierarchical_tree

    def optimize_portfolio(self, returns: pd.DataFrame, target_returns: float = 0.0) -> Tuple[np.ndarray, float]:
        cov_estimator = EWMARiskEstimator()
        cov_matrix = cov_estimator.fit(returns)

        W, _ = self._quasidiagonalization(cov_matrix)

        weights = np.sum(W, axis=1)
        weights /= np.sum(weights)

        portfolio_variance = np.dot(weights, cov_matrix.values @ weights)

        return weights, portfolio_variance