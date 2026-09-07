import numpy as np
import pandas as pd

class EWMARiskEstimator:

    def __init__(self, lambda_decay: float = 0.94):
        self.lambda_decay = lambda_decay

    def fit(self, returns: pd.DataFrame) -> pd.DataFrame:

        weights = self.lambda_decay

        cov_matrix = returns.cov()

        for i in range(len(returns)):
            r = returns.iloc[i].values

            rank_one_update = (1 - weights) * np.outer(r, r)

            cov_matrix = weights * cov_matrix + rank_one_update

        return cov_matrix

    def get_covariance_matrix(self, returns: pd.DataFrame) -> pd.DataFrame:
        return self.fit(returns)