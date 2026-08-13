# src/core/hrp.py
import numpy as np
import pandas as pd
from typing import Tuple, List
import scipy.linalg as la
from .risk_estimators import EWMARiskEstimator
class HierarchicalRiskParity:
    """
    Implements the Hierarchical Risk Parity (HRP) algorithm for portfolio optimization.
    """

    def __init__(self, min_var: float = 1e-6):
        """
        Initialize the HRP optimizer.

        Args:
            min_var: Minimum variance threshold for the regularization parameter in quasidiagonalization.
        """
        self.min_var = min_var

    def _quasidiagonalization(self, cov_matrix: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs the Quasidiagonalization algorithm to fix ill-conditioned matrices.

        Args:
            cov_matrix: A DataFrame of the covariance matrix.

        Returns:
            A tuple (W, hierarchical_tree) where:
            - W: The transformation matrix (n_assets x n_assets).
            - hierarchical_tree: A list of tuples representing the hierarchical clusters.
        """
        cov = cov_matrix.values
        n = cov.shape[0]

        # Standardize the covariance matrix (scale by the diagonal elements)
        d = np.sqrt(np.diag(cov))
        D_inv = np.diag(1 / d)
        corr = D_inv @ cov @ D_inv

        # Initialize the matrix to be diagonalized
        A = corr.copy()
        W = np.eye(n)
        hierarchical_tree = []

        # Build the hierarchical tree and transformation matrix
        # We will use the algorithm from "Hierarchical Portfolio Construction" by N. G. Avramov et al.
        # For simplicity, we implement a direct method to quasidiagonalize a covariance matrix.

        # Step 1: Find the minimum variance portfolio (MVP)
        # The MVP weights are proportional to the inverse of the covariance matrix times ones vector
        ones = np.ones(n)
        mvp_weights = np.linalg.solve(cov, ones)
        mvp_weights /= np.sum(mvp_weights)

        # Step 2: Calculate the variance of the MVP
        mvp_variance = np.dot(mvp_weights, cov @ mvp_weights)

        # Step 3: If the matrix is well-conditioned (mvp_variance > min_var), use it as the transformation matrix
        if mvp_variance > self.min_var:
            W = np.linalg.inv(np.sqrt(mvp_variance) * np.eye(n) + cov - mvp_variance * np.outer(mvp_weights, mvp_weights))
            return W, hierarchical_tree

        # Step 4: Otherwise, find the minimum variance portfolio of the sub-matrix
        # Remove one asset at a time and find the MVP of the remaining assets
        for i in range(n):
            indices = [j for j in range(n) if j != i]
            sub_cov = cov[np.ix_(indices, indices)]
            sub_ones = np.ones(len(indices))
            sub_mvp_weights = np.linalg.solve(sub_cov, sub_ones)
            sub_mvp_weights /= np.sum(sub_mvp_weights)

            sub_variance = np.dot(sub_mvp_weights, sub_cov @ sub_mvp_weights)

            if sub_variance > self.min_var:
                # Create a block diagonal matrix for the transformation
                W = np.eye(n)
                W[indices, indices] = sub_mvp_weights
                return W, hierarchical_tree

        # Fallback: Use the simple identity matrix transformation
        return np.eye(n), hierarchical_tree

    def _recursive_bisection(self, W: np.ndarray, hierarchical_tree: List) -> List:
        """
        Performs the recursive bisection algorithm to build a hierarchical tree.

        Args:
            W: The transformation matrix.
            hierarchical_tree: The hierarchical tree being built.

        Returns:
            A list of tuples representing the hierarchical clusters.
        """
        n = W.shape[0]

        if n == 1:
            return hierarchical_tree

        # Split the matrix into two halves
        half = n // 2
        left = W[:half, :half]
        right = W[half:, half:]

        # Recursively build the left and right sub-trees
        left_tree = self._recursive_bisection(left, hierarchical_tree)
        right_tree = self._recursive_bisection(right, hierarchical_tree)

        # Combine the left and right trees
        hierarchical_tree.append((left_tree, right_tree))

        return hierarchical_tree

    def optimize_portfolio(self, returns: pd.DataFrame, target_returns: float = 0.0) -> Tuple[np.ndarray, float]:
        """
        Optimizes the portfolio using the Hierarchical Risk Parity algorithm.

        Args:
            returns: A DataFrame of log returns (time steps x assets).
            target_returns: The target returns for the portfolio (default is 0.0).

        Returns:
            A tuple (weights, portfolio_variance) where:
            - weights: An array of portfolio weights (n_assets,).
            - portfolio_variance: The variance of the portfolio.
        """
        # Step 1: Compute the covariance matrix
        cov_estimator = EWMARiskEstimator()
        cov_matrix = cov_estimator.fit(returns)

        # Step 2: Quasidiagonalize the covariance matrix
        W, _ = self._quasidiagonalization(cov_matrix)

        # Step 3: Get the weights from the transformed matrix
        # The weights are the columns of the transformation matrix normalized to sum to 1
        weights = np.sum(W, axis=1)
        weights /= np.sum(weights)

        # Step 4: Calculate the portfolio variance
        portfolio_variance = np.dot(weights, cov_matrix.values @ weights)

        return weights, portfolio_variance