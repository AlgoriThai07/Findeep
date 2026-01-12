# Why our XGB looked perfect—and how we fixed it

- **Lag leakage**: Features lag_1/lag_2 + same-quarter target = trivial reconstruction, not forecasting.
- **Time matters**: Switched from random splits to chronological train/val to respect causality.
- **Next-quarter shift**: Target now y_next = revenue at t+1, features from t.
- **Backtest validation**: Holdout quarters confirm real predictive power, not in-sample fit.

**Total: 54 words**
