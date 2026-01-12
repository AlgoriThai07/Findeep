#!/usr/bin/env python3
"""
run_proper_backtest.py
End-to-end script demonstrating proper next-quarter forecasting with backtest validation.

Usage:
    python run_proper_backtest.py
"""
import pandas as pd
import yaml
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from supervised_prep import make_supervised_next_quarter, time_split
from train_next_quarter import train_xgb_next_quarter
from backtest_utils import build_test_rows, assert_no_leakage, run_backtest
from save_predictions import save_predictions_parquet
from plot_backtest import plot_backtest_bar


def main():
    # Load config
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    
    data_path = cfg.get('data_path', 'filtered_combined.json')
    kpi = cfg.get('target_kpi', 'Revenues')
    
    print("=" * 60)
    print("PROPER NEXT-QUARTER FORECASTING WORKFLOW")
    print("=" * 60)
    
    # 1. Load raw data
    print("\n1. Loading raw data...")
    df_raw = pd.read_json(data_path)
    df_raw['Date'] = pd.to_datetime(df_raw['Date'], errors='coerce')
    df_raw = df_raw.sort_values(['Company', 'Date'])
    print(f"   Loaded {len(df_raw)} rows from {data_path}")
    
    # 2. Create supervised dataset (features → y_next)
    print("\n2. Creating supervised next-quarter dataset...")
    df_supervised = make_supervised_next_quarter(df_raw, kpi=kpi)
    print(f"   Created {len(df_supervised)} supervised samples")
    print(f"   Columns: {list(df_supervised.columns)}")
    
    # 3. Time-based split
    print("\n3. Time-based train/validation split (80/20)...")
    train_df, val_df = time_split(df_supervised, frac=0.8)
    print(f"   Train: {len(train_df)} rows")
    print(f"   Val: {len(val_df)} rows")
    
    # 4. Train model
    print("\n4. Training XGBoost for next-quarter prediction...")
    # Start with minimal viable features
    feature_cols = ['lag_1', 'month', 'year']
    
    # Add lag_2 and derived if good coverage (>50%)
    optional = ['lag_2', 'diff_1', 'growth_1', 'mean_2']
    for col in optional:
        if col in train_df.columns and train_df[col].notna().sum() > len(train_df) * 0.5:
            feature_cols.append(col)
    
    # Add exogenous if very good coverage (>70%)
    optional_exog = ['OperatingExpenses', 'GrossProfit', 'Assets', 'StockholdersEquity']
    for col in optional_exog:
        if col in train_df.columns and train_df[col].notna().sum() > len(train_df) * 0.7:
            feature_cols.append(col)
    
    print(f"   Using features: {feature_cols}")
    
    model = train_xgb_next_quarter(train_df, val_df, feature_cols)
    print("   Model trained successfully.")
    
    # 5. Backtest: predict a specific future quarter
    print("\n5. Running backtest (train <= 2024-03-31, predict 2024-06-30)...")
    train_end_date = '2024-03-31'
    test_date = '2024-06-30'
    
    # Pass raw df for building test rows, supervised df for training
    comparison = run_backtest(
        df_full=df_supervised,
        df_raw=df_raw,
        train_end_date=train_end_date,
        test_date=test_date,
        feature_cols=feature_cols,
        kpi=kpi
    )
    
    if not comparison.empty:
        print("\nBacktest comparison:")
        print(comparison.to_string(index=False))
        
        # 6. Save predictions
        print("\n6. Saving backtest predictions...")
        os.makedirs('artifacts', exist_ok=True)
        save_predictions_parquet(comparison, 'artifacts/backtest_predictions.parquet')
        
        # 7. Plot
        print("\n7. Creating backtest visualization...")
        os.makedirs('slide_assets', exist_ok=True)
        plot_backtest_bar(comparison, 'slide_assets/backtest_comparison.png')
    else:
        print("\nNo backtest results to save or plot.")
    
    # 8. Leakage check
    print("\n8. Verifying no data leakage...")
    train_subset = df_supervised[df_supervised['Date'] <= pd.to_datetime(train_end_date)]
    test_rows = build_test_rows(df_raw, test_date, kpi=kpi)
    
    if not test_rows.empty:
        assert_no_leakage(train_subset, test_rows, train_end_date, test_date)
    
    print("\n" + "=" * 60)
    print("WORKFLOW COMPLETE")
    print("=" * 60)
    print("\nOutputs:")
    print("  - artifacts/backtest_predictions.parquet")
    print("  - slide_assets/backtest_comparison.png")
    print("  - docs/XGB_LEAKAGE_EXPLANATION.md")
    print("  - docs/XGB_LEAKAGE_SLIDE.md")
    print("\nNext steps:")
    print("  - Review docs/XGB_LEAKAGE_EXPLANATION.md")
    print("  - Check slide_assets/backtest_comparison.png")
    print("  - Run additional backtests for other quarters")


if __name__ == "__main__":
    main()
