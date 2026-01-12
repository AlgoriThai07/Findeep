# Why XGBoost Revenue Predictions Nearly Match Actuals

## The Problem: Data Leakage

Your XGBoost model achieves near-perfect accuracy because it's reconstructing the current quarter's revenue rather than forecasting the next quarter. The primary culprits are:

1. **Lag Feature Leakage**: Features like `lag_1` and `lag_2` contain revenue from t-1 and t-2, but your target is revenue at time t. Since revenue typically shows autocorrelation, predicting the current quarter from the previous quarter is trivial—it's essentially memorization.

2. **Same-Quarter Target**: You're training on rows where features and target come from the same historical period. The model learns "if lag_1 ≈ X, then current revenue ≈ X × (1 + small adjustment)"—this is in-sample reconstruction, not true forecasting.

3. **Missing Holdout Validation**: Without a proper time-based split, your validation set contains information from periods the model has already seen indirectly through feature engineering.

## Why This Matters

True forecasting requires predicting **unseen future quarters** using only **past information**. A backtest with a proper holdout quarter (e.g., train on data ≤ 2022-Q4, predict 2023-Q1) reveals the model's real predictive power.

## How to Fix It

- **Shift target to t+1**: Create `y_next = Revenues.shift(-1)` per company so features at time t predict revenue at t+1.
- **Time-based split**: Use chronological splits (first 80% for train, last 20% for validation) instead of random shuffling.
- **Avoid future info**: Ensure test features only use data strictly before the prediction date; never leak actuals from the target period.
