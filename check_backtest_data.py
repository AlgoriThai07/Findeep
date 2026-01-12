"""Check why backtest can't build test rows"""
import pandas as pd
from src.supervised_prep import make_supervised_next_quarter

# Load data
df_raw = pd.read_parquet('artifacts/prepared_timeseries.parquet')
print(f"Raw data: {df_raw.shape}")
print(f"\nRevenues non-null count: {df_raw['Revenues'].notna().sum()} / {len(df_raw)}")

# Make supervised
df_sup = make_supervised_next_quarter(df_raw, 'Revenues')
print(f"\nSupervised data: {df_sup.shape}")

# Check before test date
test_date = pd.Timestamp('2024-06-30')
train_end = pd.Timestamp('2024-03-31')

print(f"\n{'='*60}")
print(f"Test date: {test_date}, Train end: {train_end}")
print(f"{'='*60}")

for company in df_sup['Company'].unique():
    g = df_sup[df_sup['Company'] == company].sort_values('Date')
    
    # Raw data before test date
    g_raw = df_raw[df_raw['Company'] == company].sort_values('Date')
    past_raw = g_raw[g_raw['Date'] < test_date]
    past_raw_valid = past_raw[past_raw['Revenues'].notna()]
    
    # Supervised data before train end
    past_sup = g[g['Date'] <= train_end]
    
    print(f"\n{company}:")
    print(f"  Raw rows before test_date ({test_date}): {len(past_raw)}")
    print(f"  Raw rows with non-null Revenues: {len(past_raw_valid)}")
    print(f"  Supervised rows ≤ train_end ({train_end}): {len(past_sup)}")
    
    if len(past_raw_valid) >= 2:
        print(f"  ✓ Has ≥2 past values for lag features")
        print(f"    Last 2 valid dates: {past_raw_valid['Date'].tail(2).tolist()}")
    else:
        print(f"  ✗ Insufficient valid Revenues before test date")
