"""
models.py
Forecasting models for Toronto Island Ferry demand.
Implements:
1. Naive persistence baseline
2. Moving Average baseline
3. Regularized Linear Regression (Ridge + StandardScaler)
4. Random Forest Regressor
5. Gradient Boosting Regressor (HistGradientBoosting)
6. XGBoost Regressor (with histogram tree method)
"""

import logging
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NaiveBaseline:
    """Predicts future demand using the most recent observed demand at time t."""
    def __init__(self, target_prefix: str = 'sales'):
        self.target_prefix = target_prefix
        self.last_col = f'{target_prefix}_lag_1'

    def fit(self, X: pd.DataFrame, y: pd.Series):
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.last_col in X.columns:
            preds = X[self.last_col].values
        else:
            preds = np.zeros(len(X))
        return np.clip(preds, 0, None)

class MovingAverageBaseline:
    """Predicts future demand using rolling 1-hour average (4 steps)."""
    def __init__(self, target_prefix: str = 'sales', window_col: Optional[str] = None):
        self.target_prefix = target_prefix
        self.col = window_col or f'{target_prefix}_roll_mean_4'

    def fit(self, X: pd.DataFrame, y: pd.Series):
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.col in X.columns:
            preds = X[self.col].values
        else:
            preds = np.zeros(len(X))
        return np.clip(preds, 0, None)

class ScaledRidge:
    """Linear regression with L2 regularization and feature standardization."""
    def __init__(self, alpha: float = 100.0, random_state: int = 42):
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=alpha, random_state=random_state))
        ])

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = self.pipeline.predict(X)
        return np.clip(preds, 0, None)

class FastRandomForest:
    """Random Forest regressor tuned for accurate and fast time series training."""
    def __init__(self, n_estimators: int = 50, max_depth: int = 12, min_samples_leaf: int = 10, random_state: int = 42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_samples=0.5,
            n_jobs=-1,
            random_state=random_state
        )

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = self.model.predict(X)
        return np.clip(preds, 0, None)

class FastGradientBoosting:
    """HistGradientBoosting regressor optimized for large tabular time series."""
    def __init__(self, max_iter: int = 100, max_depth: int = 8, learning_rate: float = 0.08, random_state: int = 42):
        self.model = HistGradientBoostingRegressor(
            max_iter=max_iter,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_samples_leaf=20,
            random_state=random_state
        )

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = self.model.predict(X)
        return np.clip(preds, 0, None)

class FastXGBoost:
    """XGBoost regressor for high-efficiency gradient boosting."""
    def __init__(self, n_estimators: int = 100, max_depth: int = 6, learning_rate: float = 0.08, random_state: int = 42):
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            tree_method='hist',
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=random_state
        )

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = self.model.predict(X)
        return np.clip(preds, 0, None)

def get_model_instances(target_prefix: str = 'sales') -> Dict[str, Any]:
    """Factory creating fresh instances of all candidate models."""
    return {
        'Naive': NaiveBaseline(target_prefix=target_prefix),
        'Moving Average': MovingAverageBaseline(target_prefix=target_prefix),
        'Linear Regression': ScaledRidge(alpha=100.0),
        'Random Forest': FastRandomForest(n_estimators=50, max_depth=12),
        'Gradient Boosting': FastGradientBoosting(max_iter=100, max_depth=8),
        'XGBoost': FastXGBoost(n_estimators=100, max_depth=6)
    }
