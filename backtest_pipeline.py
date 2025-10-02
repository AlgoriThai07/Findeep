#!/usr/bin/env python3
"""
Backtesting Pipeline: Train on historical data, predict future quarter, compare with actuals
"""
import pandas as pd
import numpy as np
import sys
import subprocess
from pathlib import Path

def create_backtest_config(train_end_date, test_date, output_dir="./backtest_artifacts"):
    """
    Create a backtesting scenario:
    - Train on data up to train_end_date
    - Predict test_date
    - Compare with actual test_date results
    """
    
    # Load the full dataset
    print(f"Loading full dataset...")
    df_full = pd.read_json('./filtered_combined.json')
    df_full['Date'] = pd.to_datetime(df_full['Date'])
    
    print(f"Original data shape: {df_full.shape}")
    print(f"Date range: {df_full['Date'].min()} to {df_full['Date'].max()}")
    
    # Create train and test splits
    train_end = pd.to_datetime(train_end_date)
    test_end = pd.to_datetime(test_date)
    
    # Training data: everything up to and including train_end_date
    df_train = df_full[df_full['Date'] <= train_end].copy()
    
    # Test data: the specific test_date we want to predict
    df_test_actual = df_full[df_full['Date'] == test_end].copy()
    
    # Prediction data: training data + placeholder for test_date (for feature engineering)
    df_predict = df_train.copy()
    
    print(f"\n=== BACKTESTING SETUP ===")
    print(f"Train period: {df_train['Date'].min()} to {df_train['Date'].max()}")
    print(f"Training samples: {len(df_train)}")
    print(f"Test date: {test_end}")
    print(f"Test samples available: {len(df_test_actual)}")
    print(f"Companies in test: {df_test_actual['Company'].unique()}")
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Save training data for the pipeline
    train_file = f"{output_dir}/backtest_train_data.json"
    df_train.to_json(train_file, orient='records', date_format='iso', indent=2)
    print(f"\nSaved training data: {train_file}")
    
    # Save actual test results for comparison
    test_file = f"{output_dir}/backtest_test_actual.json" 
    df_test_actual.to_json(test_file, orient='records', date_format='iso', indent=2)
    print(f"Saved test actuals: {test_file}")
    
    return df_train, df_test_actual, output_dir

def run_backtest_pipeline(train_data_file, output_dir):
    """Run the ML pipeline on backtesting data"""
    
    # Create a temporary config that points to our training data
    config_content = f"""# Backtesting configuration
data_path: {train_data_file}
artifacts_dir: {output_dir}
target_kpi: Revenues
min_points_per_company: 3
horizon: 1
validation_fraction: 0.2
seed: 42
"""
    
    config_file = f"{output_dir}/backtest_config.yaml"
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    print(f"\n=== RUNNING BACKTEST PIPELINE ===")
    print("Step 1: Data preparation...")
    subprocess.check_call([sys.executable, "src/data_prep.py", "--config", config_file])
    
    print("Step 2: Training models...")
    subprocess.check_call([sys.executable, "src/baselines.py", "--config", config_file])
    
    print("Step 3: Generating predictions...")
    subprocess.check_call([sys.executable, "src/eval_compare.py", "--config", config_file])

def compare_predictions_with_actuals(predictions_file, actuals_file, output_dir):
    """Compare model predictions with actual results"""
    
    print(f"\n=== COMPARING PREDICTIONS VS ACTUALS ===")
    
    # Load predictions and actuals
    df_pred = pd.read_parquet(predictions_file)
    df_actual = pd.read_json(actuals_file)
    
    # Merge on Company
    df_comparison = df_pred.merge(df_actual[['Company', 'Revenues']], 
                                  on='Company', 
                                  suffixes=('_pred_base', '_actual'), 
                                  how='inner')
    
    # Calculate prediction errors
    models = ['arima_naive', 'arima_drift', 'xgb_point']
    
    # Convert all numeric columns to float to avoid type errors
    for col in ['Revenues_actual'] + models:
        if col in df_comparison.columns:
            df_comparison[col] = pd.to_numeric(df_comparison[col], errors='coerce')
    
    results = []
    for model in models:
        if model in df_comparison.columns and not df_comparison[model].isna().all():
            # Only calculate metrics for non-null predictions and actuals
            mask = df_comparison[model].notna() & df_comparison['Revenues_actual'].notna()
            if mask.sum() > 0:  # Only if we have valid data points
                predictions = df_comparison.loc[mask, model]
                actuals = df_comparison.loc[mask, 'Revenues_actual']
                
                mae = np.mean(np.abs(predictions - actuals))
                rmse = np.sqrt(np.mean((predictions - actuals)**2))
                
                results.append({
                    'model': model,
                    'MAE': mae,
                    'RMSE': rmse,
                    'MAE_billions': mae / 1e9,
                    'RMSE_billions': rmse / 1e9,
                    'valid_predictions': mask.sum()
                })
    
    # Create results dataframe
    df_results = pd.DataFrame(results)
    
    # Save detailed comparison
    comparison_file = f"{output_dir}/backtest_comparison.csv"
    df_comparison.to_csv(comparison_file, index=False)
    
    # Save metrics
    metrics_file = f"{output_dir}/backtest_metrics.csv"
    df_results.to_csv(metrics_file, index=False)
    
    print(f"\n=== BACKTEST RESULTS ===")
    print("Model Performance (Billions USD):")
    print(df_results[['model', 'MAE_billions', 'RMSE_billions']].to_string(index=False))
    
    print(f"\n=== DETAILED PREDICTIONS ===")
    display_cols = ['Company', 'Revenues_actual', 'arima_naive', 'arima_drift', 'xgb_point']
    display_cols = [col for col in display_cols if col in df_comparison.columns]
    
    df_display = df_comparison[display_cols].copy()
    # Convert to billions for readability
    for col in df_display.columns:
        if col != 'Company' and pd.api.types.is_numeric_dtype(df_display[col]):
            df_display[col] = df_display[col] / 1e9
    
    print("Values in Billions USD:")
    print(df_display.round(2).to_string(index=False))
    
    print(f"\nDetailed results saved to:")
    print(f"- {comparison_file}")
    print(f"- {metrics_file}")
    
    return df_results, df_comparison

if __name__ == "__main__":
    # Example: Train on Q1-Q2 2024, predict Q3 2024
    train_end_date = "2024-06-30"  # End of Q2 2024
    test_date = "2024-09-30"       # Q3 2024 to predict
    
    print("=== FINANCIAL FORECASTING BACKTEST ===")
    print(f"Training period: up to {train_end_date}")
    print(f"Prediction target: {test_date}")
    
    # Step 1: Create backtest data split
    df_train, df_actual, output_dir = create_backtest_config(train_end_date, test_date)
    
    if len(df_actual) == 0:
        print(f"ERROR: No actual data found for {test_date}")
        print("Available dates:", sorted(df_train['Date'].unique())[-10:])
        sys.exit(1)
    
    # Step 2: Run pipeline on training data
    train_file = f"{output_dir}/backtest_train_data.json"
    
    try:
        run_backtest_pipeline(train_file, output_dir)
        
        # Step 3: Compare predictions with actuals
        predictions_file = f"{output_dir}/predictions.parquet"
        actuals_file = f"{output_dir}/backtest_test_actual.json"
        
        results, comparison = compare_predictions_with_actuals(predictions_file, actuals_file, output_dir)
        
        print(f"\n🎉 BACKTEST COMPLETE! Check {output_dir}/ for detailed results.")
        
    except subprocess.CalledProcessError as e:
        print(f"Pipeline failed: {e}")
        print("This might be due to insufficient training data or missing dependencies.")
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Make sure you've run the original pipeline first to create the necessary source files.")