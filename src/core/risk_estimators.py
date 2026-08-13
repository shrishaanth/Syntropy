# src/core/risk_estimators.py
import numpy as np
import pandas as pd

class EWMARiskEstimator:
    """
    Computes the Exponentially Weighted Moving Average (EWMA) covariance matrix.
    Uses lambda = 0.94 as the default decay factor, as recommended by J.P. Morgan.
    """

    def __init__(self, lambda_decay: float = 0.94):
        """
        Initialize the EWMA risk estimator.

        Args:
            lambda_decay: The decay factor for the EWMA model. Default is 0.94.
        """
        self.lambda_decay = lambda_decay

    def fit(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Fits the EWMA model to the returns data and returns the covariance matrix.

        Args:
            returns: A DataFrame of log returns (time steps x assets).

        Returns:
            A pandas DataFrame containing the EWMA covariance matrix.
        """
        # Calculate the EWMA covariance matrix
        # The formula: cov_t = lambda * cov_{t-1} + (1 - lambda) * r_t * r_t^T
        # We can compute this efficiently using pandas.ewm
        # Note: pandas uses alpha = 1 - lambda for the weighting

        weights = self.lambda_decay

        # Initialize covariance matrix with the first day's sample covariance
        cov_matrix = returns.cov()

        # Apply EWMA recursively
        # We iterate over time steps (rows) in reverse to apply the recursion
        for i in range(len(returns)):
            # Get the returns vector for this time step
            r = returns.iloc[i].values

            # Calculate the rank-1 update: (1 - lambda) * r * r^T
            rank_one_update = (1 - weights) * np.outer(r, r)

            # Apply the update to the covariance matrix
            cov_matrix = weights * cov_matrix + rank_one_update

        return cov_matrix

    def get_covariance_matrix(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Convenience method to get the EWMA covariance matrix.

        Args:
            returns: A DataFrame of log returns (time steps x assets).

        Returns:
            A pandas DataFrame containing the EWMA covariance matrix.
        """
        return self.fit(returns)