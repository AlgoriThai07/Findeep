"""
save_predictions.py
Safe parquet append with deduplication.
"""
import pandas as pd
import os


def save_predictions_parquet(df: pd.DataFrame, path: str):
    """
    Save predictions to parquet, appending if file exists and deduplicating by (Company, Date).
    
    Args:
        df: DataFrame with at minimum ['Company', 'Date', 'xgb_point']
               Optional: ['arima_naive', 'arima_drift', 'pi_lower', 'pi_upper',
                          'year', 'quarter', 'month', 'Revenues']
        path: Output parquet file path
    """
    # Ensure required columns
    if 'Company' not in df.columns or 'Date' not in df.columns:
        raise ValueError("df must contain 'Company' and 'Date' columns.")
    
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Select columns to save
    save_cols = ['Company', 'Date', 'xgb_point']
    optional = ['arima_naive', 'arima_drift', 'pi_lower', 'pi_upper',
                'year', 'quarter', 'month', 'Revenues']
    for col in optional:
        if col in df.columns:
            save_cols.append(col)
    
    df_to_save = df[save_cols].copy()
    
    # Append if exists
    if os.path.exists(path):
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, df_to_save], axis=0, ignore_index=True)
        # Deduplicate by (Company, Date), keeping last occurrence
        combined = combined.drop_duplicates(subset=['Company', 'Date'], keep='last')
        combined.to_parquet(path, index=False)
        print(f"Appended and deduplicated. Saved {path} ({len(combined)} rows)")
    else:
        df_to_save.to_parquet(path, index=False)
        print(f"Saved {path} ({len(df_to_save)} rows)")
