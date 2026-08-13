import sys
import pandas as pd
import numpy as np

from src.core.data_loader import DataLoaderConfig
from src.core.utils import clean_data
from src.core.risk_estimators import EWMARiskEstimator
from src.core.hrp import HierarchicalRiskParity
from src.core.backtest import Strategy

print('✅ All modules imported successfully')

# Quick functional test
np.random.seed(42)
test_data = pd.DataFrame(np.random.randn(252,5)*100+1000, index=pd.date_range('2023-01-01',periods=252), columns=['AAPL','MSFT','NVDA','JPM','XOM'])
cleaned = clean_data(test_data)
print(f'✅ Data cleaning: {len(cleaned)} days, {len(cleaned.columns)} assets')

ewma = EWMARiskEstimator()
returns = pd.DataFrame(np.random.randn(252,5), index=pd.date_range('2023-01-01',periods=252))
cov = ewma.fit(returns)
print(f'✅ EWMA covariance: {cov.shape}')

hrp = HierarchicalRiskParity()
weights, var = hrp.optimize_portfolio(returns)
print(f'✅ HRP: weights={weights.sum():.6f}, variance={var:.6f}')

mock = pd.DataFrame(np.random.randn(252,5).cumsum(axis=0)*10+100, index=pd.date_range('2023-01-01',periods=252), columns=['AAPL','MSFT','NVDA','JPM','XOM'])
mock['SPY'] = np.random.randn(252).cumsum()*5+100
strat = Strategy(train_window=126, test_window=63, benchmarks=['SPY'])
results, weights_df, costs = strat.run(mock)
print(f'✅ Strategy: {len(results)} days, {len(weights_df)} rebalances')

print('\\\\n🎉 Phase 1 COMPLETE - Ready for Phase 2!')