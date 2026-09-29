"""
data_processing.py
Data processing, validation, chronological sorting, and 15-minute grid alignment
for the Toronto Island Ferry Ticket dataset.
"""

import os
import logging
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RAW_FILE_NAME = "Toronto_Island_Ferry_Tickets.csv"
PROCESSED_FILE_NAME = "clean_15min_timeseries.parquet"
PROCESSED_CSV_NAME = "clean_15min_timeseries.csv"

def load_raw_dataset(raw_path: str) -> pd.DataFrame:
    """Loads the raw ferry ticket dataset with strict schema verification."""
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data file not found at: {raw_path}")
    
    logger.info(f"Loading raw dataset from {raw_path}...")
    df = pd.read_csv(raw_path)
    
    expected_cols = {'_id', 'Timestamp', 'Redemption Count', 'Sales Count'}
    actual_cols = set(df.columns)
    if not expected_cols.issubset(actual_cols):
        raise ValueError(f"Dataset columns mismatch. Expected at least {expected_cols}, got {actual_cols}")
    
    logger.info(f"Raw dataset loaded successfully: {len(df):,} rows, {len(df.columns)} columns.")
    return df

def audit_and_clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Performs data cleaning:
    1. Parse timestamps to datetime.
    2. Sort chronologically.
    3. Remove any duplicate timestamps (if present).
    4. Enforce non-negative counts.
    5. Align to a regular 15-minute grid.
       Missing intervals represent zero activity (periods where both sales and redemptions were 0
       and hence omitted from transactional logs, e.g., overnight terminal closure).
    """
    logger.info("Parsing timestamps and sorting chronologically...")
    df = df.copy()
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    
    invalid_ts = df['Timestamp'].isna().sum()
    if invalid_ts > 0:
        logger.warning(f"Dropping {invalid_ts} rows with invalid timestamps.")
        df = df.dropna(subset=['Timestamp'])
    
    duplicates_count = int(df.duplicated(subset=['Timestamp']).sum())
    if duplicates_count > 0:
        logger.warning(f"Found {duplicates_count} duplicate timestamps. Aggregating by sum.")
        df = df.groupby('Timestamp', as_index=False).agg({
            'Sales Count': 'sum',
            'Redemption Count': 'sum'
        })
    else:
        df = df[['Timestamp', 'Sales Count', 'Redemption Count']]
    
    df = df.sort_values('Timestamp').reset_index(drop=True)
    
    df['Sales Count'] = df['Sales Count'].clip(lower=0).astype(np.int64)
    df['Redemption Count'] = df['Redemption Count'].clip(lower=0).astype(np.int64)
    
    min_date = df['Timestamp'].min()
    max_date = df['Timestamp'].max()
    raw_row_count = len(df)
    
    logger.info(f"Building complete 15-minute frequency grid from {min_date} to {max_date}...")
    full_idx = pd.date_range(start=min_date, end=max_date, freq='15min', name='Timestamp')
    
    df_indexed = df.set_index('Timestamp')
    
    missing_intervals = len(full_idx) - len(df_indexed)
    logger.info(f"Full 15-min intervals: {len(full_idx):,}. Raw active intervals: {raw_row_count:,}. "
                f"Missing intervals: {missing_intervals:,} ({(missing_intervals/len(full_idx))*100:.2f}%).")
    
    df_clean = df_indexed.reindex(full_idx, fill_value=0).reset_index()
    
    df_clean['is_operating_hours'] = ((df_clean['Timestamp'].dt.hour >= 7) & 
                                      (df_clean['Timestamp'].dt.hour <= 23)).astype(int)
    
    metadata = {
        'raw_rows': raw_row_count,
        'clean_rows': len(df_clean),
        'min_timestamp': str(min_date),
        'max_timestamp': str(max_date),
        'missing_intervals_zero_filled': missing_intervals,
        'duplicates_removed': duplicates_count,
        'sales_total': int(df_clean['Sales Count'].sum()),
        'redemptions_total': int(df_clean['Redemption Count'].sum()),
        'sales_mean': float(df_clean['Sales Count'].mean()),
        'sales_max': int(df_clean['Sales Count'].max()),
        'redemption_mean': float(df_clean['Redemption Count'].mean()),
        'redemption_max': int(df_clean['Redemption Count'].max()),
    }
    
    logger.info("Data audit and cleaning complete.")
    return df_clean, metadata

def run_data_processing(raw_dir: str, processed_dir: str) -> pd.DataFrame:
    """End-to-end data processing entrypoint."""
    raw_file = os.path.join(raw_dir, RAW_FILE_NAME)
    os.makedirs(processed_dir, exist_ok=True)
    
    df_raw = load_raw_dataset(raw_file)
    df_clean, metadata = audit_and_clean_data(df_raw)
    
    parquet_path = os.path.join(processed_dir, PROCESSED_FILE_NAME)
    csv_path = os.path.join(processed_dir, PROCESSED_CSV_NAME)
    
    logger.info(f"Saving processed data to {parquet_path}...")
    df_clean.to_parquet(parquet_path, index=False)
    
    logger.info("Processing complete and artifacts saved.")
    return df_clean

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    raw_dir = os.path.join(project_root, "data", "raw")
    processed_dir = os.path.join(project_root, "data", "processed")
    
    df_clean = run_data_processing(raw_dir, processed_dir)
    print(df_clean.head(10))
