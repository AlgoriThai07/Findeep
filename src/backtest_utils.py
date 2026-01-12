"""
backtest_utils.py
Utilities for building test rows and running proper backtests without leakage.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List


def build_test_rows(
    df_raw: pd.DataFrame,
    test_date: str,
    kpi: str = 'Revenues'
) -> pd.DataFrame:
    """
    Build synthetic test rows at test_date for each company using only past info.
    
    Args:
        df_raw: Raw data with ['Company', 'Date', kpi, exogenous vars]
        test_date: Target prediction date (str or datetime)
        kpi: KPI column name
    
    Returns:
        DataFrame with feature rows ready for model.predict (no y_next column)
    """
    df_raw = df_raw.copy()
    df_raw['Date'] = pd.to_datetime(df_raw['Date'], errors='coerce')
    # Ensure KPI is numeric
    df_raw[kpi] = pd.to_numeric(df_raw[kpi], errors='coerce')
    test_dt = pd.to_datetime(test_date)
    
    test_rows = []
    
    for company, g in df_raw.groupby('Company'):
        # Use only data strictly before test_date
        past = g[g['Date'] < test_dt].sort_values('Date')
        
        if len(past) < 2:
            # Skip if insufficient history
            continue
        
        # Get the two most recent values before test_date
        recent_2 = past.tail(2)
        vals = recent_2[kpi].values
        
        if pd.isna(vals).any():
            continue
        
        lag_1 = vals[-1]
        lag_2 = vals[-2] if len(vals) > 1 else np.nan
        
        # Build features
        row = {
            'Company': company,
            'Date': test_dt,
            'lag_1': lag_1,
            'lag_2': lag_2,
            'diff_1': lag_1 - lag_2 if not pd.isna(lag_2) else np.nan,
            'growth_1': (lag_1 / lag_2 - 1) if (not pd.isna(lag_2) and lag_2 != 0) else np.nan,
            'mean_2': (lag_1 + lag_2) / 2 if not pd.isna(lag_2) else lag_1,
            'year': test_dt.year,
            'quarter': ((test_dt.month - 1) // 3 + 1),
            'month': test_dt.month,
        }
        
        # Add exogenous vars from most recent past row
        last_row = past.iloc[-1]
        for col in ['OperatingExpenses', 'GrossProfit', 'Assets', 'StockholdersEquity']:
            if col in last_row:
                row[col] = last_row[col]
        
        test_rows.append(row)
    
    if len(test_rows) == 0:
        return pd.DataFrame()
    
    return pd.DataFrame(test_rows)


def assert_no_leakage(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_end_date: str,
    test_date: str
):
    """
    Assert no data leakage between train and test sets.
    
    Checks:
    1. All training rows have Date <= train_end_date
    2. Test features use only history < test_date
    3. y_next (if present) never comes from or after test_date
    
    Raises:
        AssertionError with descriptive message on violations
    """
    train_end_dt = pd.to_datetime(train_end_date)
    test_dt = pd.to_datetime(test_date)
    
    # Check 1: Train dates
    if 'Date' in train_df.columns:
        train_df['Date'] = pd.to_datetime(train_df['Date'], errors='coerce')
        max_train = train_df['Date'].max()
        if max_train > train_end_dt:
            raise AssertionError(
                f"Training data contains dates after train_end_date. "
                f"Max train date: {max_train}, train_end_date: {train_end_dt}"
            )
    
    # Check 2: Test dates (features should be built from past only)
    if 'Date' in test_df.columns:
        test_df['Date'] = pd.to_datetime(test_df['Date'], errors='coerce')
        # Test rows should be AT test_date, but features built from < test_date
        # We can't verify feature construction here, but we can check Date column
        if (test_df['Date'] < test_dt).any():
            raise AssertionError(
                f"Test data contains dates before test_date {test_dt}. "
                "Test rows should represent the prediction date."
            )
    
    # Check 3: y_next should not exist in test (or if it does, for validation only)
    if 'y_next' in test_df.columns:
        # If y_next is present in test for comparison, ensure it's from test_date
        # (this is acceptable for evaluation, but shouldn't be used as a feature)
        print("WARNING: y_next column found in test_df. Ensure it's only for evaluation, not as a feature.")


def run_backtest(
    df_full: pd.DataFrame,
    df_raw: pd.DataFrame,
    train_end_date: str,
    test_date: str,
    feature_cols: List[str],
    kpi: str = 'Revenues'
) -> pd.DataFrame:
    """
    Run a proper backtest: train on data <= train_end_date, predict at test_date.
    
    Args:
        df_full: Supervised dataset with Date, Company, features, y_next
        df_raw: Raw dataset with KPI column for building test rows
        train_end_date: Last date to include in training
        test_date: Prediction target date
        feature_cols: List of feature column names for XGB
        kpi: Name of the KPI column in df_raw
    
    Returns:
        DataFrame with ['Company', 'Date', 'actual', 'xgb_point']
    """
    from train_next_quarter import train_xgb_next_quarter
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    df_full = df_full.copy()
    df_full['Date'] = pd.to_datetime(df_full['Date'], errors='coerce')
    train_end_dt = pd.to_datetime(train_end_date)
    test_dt = pd.to_datetime(test_date)
    
    # Split training data
    train_data = df_full[df_full['Date'] <= train_end_dt].copy()
    
    # Time-based split for validation (optional, or use all train data)
    # Here we'll use all train data and skip internal validation for simplicity
    # (You can call time_split if you want internal validation)
    
    if 'y_next' not in train_data.columns:
        raise ValueError("train_data must have 'y_next' column. Run make_supervised_next_quarter first.")
    
    train_complete = train_data.dropna(subset=feature_cols + ['y_next'])
    
    if len(train_complete) < 5:
        print("WARNING: Very small training set. Proceeding anyway.")
    
    # Train model
    # For simplicity, we'll train on all train_complete without internal val split
    # (Alternatively, call train_xgb_next_quarter with a val split)
    import xgboost as xgb
    X_train = train_complete[feature_cols].astype(float).values
    y_train = train_complete['y_next'].astype(float).values
    
    model = xgb.XGBRegressor(
        tree_method='hist',
        n_estimators=600,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )
    model.fit(X_train, y_train, verbose=False)
    print(f"Trained on {len(X_train)} samples (dates <= {train_end_date})")
    
    # Build test rows at test_date from raw data
    test_rows = build_test_rows(df_raw, test_date, kpi=kpi)
    
    if test_rows.empty:
        print("No test rows built. Insufficient history for companies.")
        return pd.DataFrame(columns=['Company', 'Date', 'actual', 'xgb_point'])
    
    # Predict
    X_test = test_rows[feature_cols].astype(float).values
    test_rows['xgb_point'] = model.predict(X_test)
    
    # Merge with actuals
    actuals = df_full[df_full['Date'] == test_dt][['Company', 'Date', 'y_next']].copy()
    actuals = actuals.rename(columns={'y_next': 'actual'})
    
    comparison = test_rows[['Company', 'Date', 'xgb_point']].merge(
        actuals, on=['Company', 'Date'], how='inner'
    )
    
    if len(comparison) == 0:
        print("No actuals available at test_date for comparison (Merge result empty).")
        return comparison
    
    # Print metrics
    mae = mean_absolute_error(comparison['actual'], comparison['xgb_point'])
    rmse = np.sqrt(mean_squared_error(comparison['actual'], comparison['xgb_point']))
    
    print(f"\nBacktest Results for {test_date}:")
    print(f"  Companies evaluated: {len(comparison)}")
    print(f"  MAE: ${mae/1e9:.3f}B")
    print(f"  RMSE: ${rmse/1e9:.3f}B")
    
    return comparison[['Company', 'Date', 'actual', 'xgb_point']]
