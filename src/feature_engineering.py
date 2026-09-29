"""
feature_engineering.py
Feature engineering module for short-term ferry demand forecasting.
Computes lag features, rolling statistics, calendar indicators, cyclical encodings,
and multi-horizon forecasting targets with strict temporal leakage prevention.
"""

import os
import logging
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

HORIZON_STEPS = {
    '15m': 1,
    '30m': 2,
    '1h': 4,
    '2h': 8
}

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts temporal attributes and cyclical encodings."""
    df = df.copy()
    ts = df['Timestamp']
    
    df['hour'] = ts.dt.hour
    df['minute'] = ts.dt.minute
    df['day_of_week'] = ts.dt.dayofweek
    df['day_of_month'] = ts.dt.day
    df['month'] = ts.dt.month
    df['year'] = ts.dt.year
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Peak summer ferry season (June, July, August)
    df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
    # Shoulder season (May, September)
    df['is_shoulder'] = df['month'].isin([5, 9]).astype(int)
    
    # Cyclical encodings
    # 24-hour cycle
    hour_float = df['hour'] + df['minute'] / 60.0
    df['sin_hour'] = np.sin(2 * np.pi * hour_float / 24.0)
    df['cos_hour'] = np.cos(2 * np.pi * hour_float / 24.0)
    
    # 7-day cycle
    dow_float = df['day_of_week'] + hour_float / 24.0
    df['sin_dow'] = np.sin(2 * np.pi * dow_float / 7.0)
    df['cos_dow'] = np.cos(2 * np.pi * dow_float / 7.0)
    
    # 12-month cycle
    df['sin_month'] = np.sin(2 * np.pi * (df['month'] - 1) / 12.0)
    df['cos_month'] = np.cos(2 * np.pi * (df['month'] - 1) / 12.0)
    
    return df

def add_ontario_holidays(df: pd.DataFrame) -> pd.DataFrame:
    """Flags major Ontario statutory holidays that drive high ferry tourism demand."""
    df = df.copy()
    dates = df['Timestamp'].dt.date
    
    # Generate holiday dates for 2015-2026
    holiday_dates = set()
    for yr in range(2015, 2027):
        # New Year
        holiday_dates.add(pd.Timestamp(year=yr, month=1, day=1).date())
        # Canada Day
        holiday_dates.add(pd.Timestamp(year=yr, month=7, day=1).date())
        # Christmas & Boxing Day
        holiday_dates.add(pd.Timestamp(year=yr, month=12, day=25).date())
        holiday_dates.add(pd.Timestamp(year=yr, month=12, day=26).date())
        
        # Victoria Day: Monday on or before May 24
        may24 = pd.Timestamp(year=yr, month=5, day=24)
        vic_day = may24 - pd.Timedelta(days=may24.weekday())
        holiday_dates.add(vic_day.date())
        
        # Civic Holiday: First Monday in August (huge Toronto Island weekend!)
        aug1 = pd.Timestamp(year=yr, month=8, day=1)
        civic_day = aug1 + pd.Timedelta(days=(7 - aug1.weekday()) % 7)
        holiday_dates.add(civic_day.date())
        
        # Labour Day: First Monday in September
        sep1 = pd.Timestamp(year=yr, month=9, day=1)
        labour_day = sep1 + pd.Timedelta(days=(7 - sep1.weekday()) % 7)
        holiday_dates.add(labour_day.date())
        
        # Thanksgiving: Second Monday in October
        oct1 = pd.Timestamp(year=yr, month=10, day=1)
        first_mon = oct1 + pd.Timedelta(days=(7 - oct1.weekday()) % 7)
        thanksgiving = first_mon + pd.Timedelta(days=7)
        holiday_dates.add(thanksgiving.date())
        
    df['is_holiday'] = dates.isin(holiday_dates).astype(int)
    return df

def add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes lag and rolling window features for both Sales and Redemptions.
    Strictly uses current and past observations (shift >= 0 relative to time t).
    """
    df = df.copy()
    targets = ['Sales Count', 'Redemption Count']
    
    # 15-min intervals:
    # 1=15m, 2=30m, 4=1h, 8=2h, 12=3h, 24=6h, 96=24h (same time yesterday), 672=7d (same time last week)
    lag_steps = [1, 2, 4, 8, 12, 24, 96, 672]
    
    for col in targets:
        prefix = 'sales' if col == 'Sales Count' else 'redemption'
        
        # Lag features
        for lag in lag_steps:
            df[f'{prefix}_lag_{lag}'] = df[col].shift(lag)
            
        # Rolling features (shift(1) ensures no lookahead of current step)
        s1 = df[col].shift(1)
        df[f'{prefix}_roll_mean_4'] = s1.rolling(window=4, min_periods=1).mean()
        df[f'{prefix}_roll_mean_8'] = s1.rolling(window=8, min_periods=1).mean()
        df[f'{prefix}_roll_mean_24'] = s1.rolling(window=24, min_periods=1).mean()
        df[f'{prefix}_roll_mean_96'] = s1.rolling(window=96, min_periods=1).mean()
        df[f'{prefix}_roll_std_8'] = s1.rolling(window=8, min_periods=1).std().fillna(0)
        df[f'{prefix}_roll_max_8'] = s1.rolling(window=8, min_periods=1).max()
        df[f'{prefix}_roll_min_8'] = s1.rolling(window=8, min_periods=1).min()
    
    # Cross-target features (sales leading redemption queues)
    df['sales_to_redemption_ratio_lag1'] = (
        (df['sales_lag_1'] + 1.0) / (df['redemption_lag_1'] + 1.0)
    )
    df['sales_redemption_diff_lag1'] = df['sales_lag_1'] - df['redemption_lag_1']
    df['sales_redemption_diff_lag4'] = df['sales_lag_4'] - df['redemption_lag_4']
    
    return df

def add_future_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates target columns for multi-horizon forecasting:
    - target_sales_h1 (15m ahead)
    - target_sales_h2 (30m ahead)
    - target_sales_h4 (1h ahead)
    - target_sales_h8 (2h ahead)
    - target_redemption_h1 (15m ahead)
    - target_redemption_h2 (30m ahead)
    - target_redemption_h4 (1h ahead)
    - target_redemption_h8 (2h ahead)
    """
    df = df.copy()
    for name, step in HORIZON_STEPS.items():
        df[f'target_sales_{name}'] = df['Sales Count'].shift(-step)
        df[f'target_redemption_{name}'] = df['Redemption Count'].shift(-step)
    return df

def get_feature_columns() -> List[str]:
    """Returns the standardized list of predictor feature names."""
    features = [
        'hour', 'minute', 'day_of_week', 'day_of_month', 'month', 'is_weekend',
        'is_summer', 'is_shoulder', 'is_holiday', 'is_operating_hours',
        'sin_hour', 'cos_hour', 'sin_dow', 'cos_dow', 'sin_month', 'cos_month',
        'sales_to_redemption_ratio_lag1', 'sales_redemption_diff_lag1', 'sales_redemption_diff_lag4'
    ]
    lag_steps = [1, 2, 4, 8, 12, 24, 96, 672]
    for prefix in ['sales', 'redemption']:
        for lag in lag_steps:
            features.append(f'{prefix}_lag_{lag}')
        features.extend([
            f'{prefix}_roll_mean_4',
            f'{prefix}_roll_mean_8',
            f'{prefix}_roll_mean_24',
            f'{prefix}_roll_mean_96',
            f'{prefix}_roll_std_8',
            f'{prefix}_roll_max_8',
            f'{prefix}_roll_min_8'
        ])
    return features

def build_features_pipeline(clean_timeseries_path: str, output_path: str) -> pd.DataFrame:
    """Runs end-to-end feature creation and saves engineered dataset."""
    logger.info(f"Loading clean time series from {clean_timeseries_path}...")
    df = pd.read_parquet(clean_timeseries_path)
    
    logger.info("Adding temporal and holiday features...")
    df = add_temporal_features(df)
    df = add_ontario_holidays(df)
    
    logger.info("Computing lag and rolling features...")
    df = add_lag_and_rolling_features(df)
    
    logger.info("Computing multi-horizon targets...")
    df = add_future_targets(df)
    
    # We drop the first 672 rows (1 week warm-up period for lag_672) so all lags are fully populated
    # And drop the last 8 rows where target_h8 is NaN (at the very end of the 10-year timeline)
    warmup_rows = 672
    max_h = 8
    df_valid = df.iloc[warmup_rows:-max_h].copy().reset_index(drop=True)
    
    logger.info(f"Feature dataset shape after warm-up trimming: {df_valid.shape}")
    logger.info(f"Saving feature dataset to {output_path}...")
    df_valid.to_parquet(output_path, index=False)
    
    return df_valid

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    clean_path = os.path.join(project_root, "data", "processed", "clean_15min_timeseries.parquet")
    output_path = os.path.join(project_root, "data", "processed", "features_dataset.parquet")
    
    df_features = build_features_pipeline(clean_path, output_path)
    print("Features engineered successfully.")
    print("Columns sample:", list(df_features.columns)[:20])
