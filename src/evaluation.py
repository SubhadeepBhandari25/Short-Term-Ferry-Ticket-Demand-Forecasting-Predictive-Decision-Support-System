"""
evaluation.py
Comprehensive evaluation metrics, time-series splitting, peak miss rate analysis,
and empirical prediction interval calculation for Toronto Island ferry demand forecasting.
"""

import logging
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def split_chronological(df: pd.DataFrame, 
                        val_start: str = '2024-07-01', 
                        test_start: str = '2025-06-01') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological time series splitting avoiding temporal leakage.
    - Train: beginning to val_start (~9 years)
    - Validation: val_start to test_start (~11 months)
    - Test Holdout: test_start to end (~7 months of recent 2025 demand)
    """
    ts = pd.to_datetime(df['Timestamp'])
    val_dt = pd.to_datetime(val_start)
    test_dt = pd.to_datetime(test_start)
    
    train_mask = ts < val_dt
    val_mask = (ts >= val_dt) & (ts < test_dt)
    test_mask = ts >= test_dt
    
    df_train = df[train_mask].copy().reset_index(drop=True)
    df_val = df[val_mask].copy().reset_index(drop=True)
    df_test = df[test_mask].copy().reset_index(drop=True)
    
    logger.info(f"Chronological split sizes: Train={len(df_train):,} ({len(df_train)/len(df)*100:.1f}%), "
                f"Val={len(df_val):,} ({len(df_val)/len(df)*100:.1f}%), "
                f"Test={len(df_test):,} ({len(df_test)/len(df)*100:.1f}%)")
    return df_train, df_val, df_test

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, peak_threshold: float = 240.0) -> Dict[str, float]:
    """
    Computes real metrics:
    - MAE
    - RMSE
    - WAPE (Weighted Absolute Percentage Error)
    - sMAPE (Symmetric Mean Absolute Percentage Error)
    - MAPE_positive (MAPE calculated exclusively on y_true > 0)
    - Peak Miss Rate (fraction of actual peak events with >30% underprediction)
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    y_p = np.clip(y_p, 0, None)
    
    # Absolute errors
    abs_errors = np.abs(y_t - y_p)
    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean((y_t - y_p) ** 2)))
    
    # WAPE = sum(|y - y_hat|) / sum(y)
    sum_yt = np.sum(y_t)
    wape = float(np.sum(abs_errors) / sum_yt * 100.0) if sum_yt > 0 else 0.0
    
    # sMAPE = 100% * mean( 2*|y - y_hat| / (|y| + |y_hat| + 1e-6) )
    denom = (np.abs(y_t) + np.abs(y_p)) / 2.0
    # when both are zero, error is zero
    zero_mask = (y_t == 0) & (y_p == 0)
    smape_vals = np.zeros_like(y_t)
    nonzero = ~zero_mask
    smape_vals[nonzero] = abs_errors[nonzero] / np.maximum(denom[nonzero], 1e-6) * 100.0
    smape = float(np.mean(smape_vals))
    
    # Non-zero MAPE
    pos_mask = y_t > 0
    if np.sum(pos_mask) > 0:
        mape_pos = float(np.mean(abs_errors[pos_mask] / y_t[pos_mask]) * 100.0)
    else:
        mape_pos = 0.0
        
    # Peak Miss Rate:
    # A peak event is when actual demand >= peak_threshold (e.g. 95th percentile).
    # A "miss" is defined as underpredicting by more than 30% of the actual demand.
    peak_mask = y_t >= peak_threshold
    peak_count = int(np.sum(peak_mask))
    if peak_count > 0:
        # underprediction occurs when y_pred < 0.70 * y_true
        underpredicted_peaks = np.sum((y_p[peak_mask] < 0.70 * y_t[peak_mask]))
        peak_miss_rate = float(underpredicted_peaks / peak_count * 100.0)
    else:
        peak_miss_rate = 0.0
        
    return {
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'WAPE': round(wape, 2),
        'sMAPE': round(smape, 2),
        'MAPE_pos': round(mape_pos, 2),
        'Peak_Miss_Rate': round(peak_miss_rate, 2),
        'Peak_Count': peak_count,
        'Peak_Threshold': round(float(peak_threshold), 1)
    }

def compute_prediction_interval_margin(residuals: np.ndarray, coverage: float = 0.90) -> float:
    """
    Computes the empirical residual margin for a given coverage level (e.g., 0.90 for 90% interval).
    Interval is [y_pred - margin, y_pred + margin] clipped at zero.
    """
    abs_res = np.abs(residuals)
    margin = float(np.percentile(abs_res, coverage * 100.0))
    return round(margin, 2)

if __name__ == '__main__':
    y_test = np.array([0, 10, 50, 200, 300, 500])
    y_pred = np.array([0, 12, 45, 180, 250, 320])
    m = compute_metrics(y_test, y_pred, peak_threshold=200.0)
    print("Metrics test:", m)
