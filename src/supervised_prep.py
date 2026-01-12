"""
supervised_prep.py
Refactored feature preparation for true next-quarter forecasting (t+1).
"""
import pandas as pd
import numpy as np


def make_supervised_next_quarter(df: pd.DataFrame, kpi: str = 'Revenues') -> pd.DataFrame:
    """
    Transform raw data into supervised learning format where features at time t
    predict the KPI at time t+1.
    
    Args:
        df: DataFrame with at least ['Company', 'Date', kpi, exogenous vars]
        kpi: Target KPI column name
    
    Returns:
        DataFrame with features from current quarter and y_next = next quarter KPI.
        Columns: ['Company','Date','year','quarter','month',
                  'lag_1','lag_2','diff_1','growth_1','mean_2',
                  'OperatingExpenses','GrossProfit','Assets','StockholdersEquity',
                  'y_next']
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Ensure KPI and exogenous are numeric
    numeric_cols = [kpi, 'OperatingExpenses', 'GrossProfit', 'Assets', 'StockholdersEquity']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    frames = []
    for company, g in df.groupby('Company'):
        g = g.sort_values('Date').copy()
        
        # Current-quarter features (lags from the past)
        g['lag_1'] = g[kpi].shift(1)
        g['lag_2'] = g[kpi].shift(2)
        g['diff_1'] = g[kpi] - g['lag_1']
        g['growth_1'] = (g[kpi] / g['lag_1'] - 1).replace([np.inf, -np.inf], np.nan)
        g['mean_2'] = (g[kpi] + g['lag_1']) / 2
        
        # Time features
        g['month'] = g['Date'].dt.month
        g['year'] = g['Date'].dt.year
        g['quarter'] = ((g['Date'].dt.month - 1) // 3 + 1).astype(int)
        
        # Target: next quarter's KPI
        g['y_next'] = g[kpi].shift(-1)
        
        frames.append(g)
    
    result = pd.concat(frames, axis=0, ignore_index=True)
    
    # Drop rows where y_next is NaN (last row per company has no future)
    result = result.dropna(subset=['y_next'])
    
    # Select final columns
    cols = [
        'Company', 'Date', 'year', 'quarter', 'month',
        'lag_1', 'lag_2', 'diff_1', 'growth_1', 'mean_2',
        'OperatingExpenses', 'GrossProfit', 'Assets', 'StockholdersEquity',
        'y_next'
    ]
    # Keep only columns that exist
    cols = [c for c in cols if c in result.columns]
    
    return result[cols]


def time_split(df: pd.DataFrame, frac: float = 0.8):
    """
    Time-based train/validation split (no shuffling).
    
    Args:
        df: DataFrame with 'Date' column
        frac: Fraction of data for training (chronologically first rows)
    
    Returns:
        (train_df, val_df): Disjoint DataFrames respecting chronological order
    """
    df = df.sort_values('Date').reset_index(drop=True)
    split_idx = int(len(df) * frac)
    
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()
    
    return train_df, val_df
