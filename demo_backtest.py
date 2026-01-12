"""
demo_backtest.py
Simple demonstration of the proper next-quarter forecasting workflow.
Shows the leakage problem and the fix with a synthetic example.
"""
import pandas as pd
import numpy as np

print("=" * 70)
print("DEMONSTRATION: Why XGBoost≈actuals and how to fix it")
print("=" * 70)

# Create synthetic quarterly revenue data for 2 companies
dates = pd.date_range('2020-03-31', periods=16, freq='Q')
np.random.seed(42)

data = []
for company in ['CompanyA', 'CompanyB']:
    base = 10.0 if company == 'CompanyA' else 15.0
    for i, d in enumerate(dates):
        revenue = base + i * 0.5 + np.random.normal(0, 0.3)
        data.append({
            'Company': company,
            'Date': d,
            'Revenues': revenue
        })

df = pd.DataFrame(data)
print("\n1. Sample data (first 8 rows):")
print(df.head(8)[['Company', 'Date', 'Revenues']].to_string(index=False))

# WRONG WAY: Same-quarter features → same-quarter target
print("\n2. WRONG WAY: Features from t predict target at t")
df_wrong = df.copy()
df_wrong = df_wrong.sort_values(['Company', 'Date'])
for company, g in df_wrong.groupby('Company'):
    idx = g.index
    df_wrong.loc[idx, 'lag_1'] = g['Revenues'].shift(1)
    df_wrong.loc[idx, 'target'] = g['Revenues']  # SAME QUARTER!

df_wrong = df_wrong.dropna(subset=['lag_1', 'target'])
print(df_wrong[['Company', 'Date', 'lag_1', 'target']].head(6).to_string(index=False))
print("\n   → Problem: lag_1 ≈ target (autocorrelation), model just memorizes!")

# RIGHT WAY: Features from t predict target at t+1
print("\n3. RIGHT WAY: Features from t predict target at t+1")
df_right = df.copy()
df_right = df_right.sort_values(['Company', 'Date'])
for company, g in df_right.groupby('Company'):
    idx = g.index
    df_right.loc[idx, 'lag_1'] = g['Revenues'].shift(1)
    df_right.loc[idx, 'y_next'] = g['Revenues'].shift(-1)  # NEXT QUARTER!

df_right = df_right.dropna(subset=['lag_1', 'y_next'])
print(df_right[['Company', 'Date', 'lag_1', 'y_next']].head(6).to_string(index=False))
print("\n   → Correct: lag_1 at t used to forecast y_next at t+1")

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)
print("""
✓ Created explanation doc:     docs/XGB_LEAKAGE_EXPLANATION.md
✓ Created slide bullets:        docs/XGB_LEAKAGE_SLIDE.md
✓ Created supervised prep code: src/supervised_prep.py
✓ Created training code:        src/train_next_quarter.py
✓ Created backtest utils:       src/backtest_utils.py
✓ Created save/plot code:       src/save_predictions.py, src/plot_backtest.py
✓ Created runner:               run_proper_backtest.py

Next steps:
1. Review docs/XGB_LEAKAGE_EXPLANATION.md for full explanation
2. Review docs/XGB_LEAKAGE_SLIDE.md for slide bullets
3. Use src/supervised_prep.make_supervised_next_quarter() for t+1 targets
4. Use src/supervised_prep.time_split() for chronological train/val splits
5. Use src/backtest_utils.run_backtest() for proper holdout evaluation
""")
