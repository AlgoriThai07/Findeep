"""
train_next_quarter.py
Train XGBoost for true next-quarter forecasting with time-based validation.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error


def train_xgb_next_quarter(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list
) -> xgb.XGBRegressor:
    """
    Train XGBoost regressor on y_next target with time-based validation.
    
    Args:
        train_df: Training data with 'y_next' column
        val_df: Validation data with 'y_next' column
        feature_cols: List of feature column names
    
    Returns:
        Trained XGBRegressor model
    """
    # Filter to complete cases
    train_complete = train_df.dropna(subset=feature_cols + ['y_next']).copy()
    val_complete = val_df.dropna(subset=feature_cols + ['y_next']).copy()
    
    if len(train_complete) == 0:
        raise ValueError("No complete training samples available after dropping NaNs.")
    
    X_train = train_complete[feature_cols].astype(float).values
    y_train = train_complete['y_next'].astype(float).values
    
    print(f"Training samples: {len(X_train)}")
    
    # Handle small datasets gracefully
    if len(train_complete) < 5:
        print("WARNING: Very small dataset. Training without eval_set.")
        model = xgb.XGBRegressor(
            tree_method='hist',
            n_estimators=200,
            max_depth=3,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42
        )
        model.fit(X_train, y_train, verbose=False)
    else:
        model = xgb.XGBRegressor(
            tree_method='hist',
            n_estimators=600,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42
        )
        
        if len(val_complete) > 0:
            X_val = val_complete[feature_cols].astype(float).values
            y_val = val_complete['y_next'].astype(float).values
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            
            # Print validation metrics
            y_pred_val = model.predict(X_val)
            mae = mean_absolute_error(y_val, y_pred_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
            print(f"Validation samples: {len(X_val)}")
            print(f"Validation MAE: ${mae/1e9:.3f}B")
            print(f"Validation RMSE: ${rmse/1e9:.3f}B")
        else:
            print("WARNING: No validation samples. Training without eval_set.")
            model.fit(X_train, y_train, verbose=False)
    
    return model
