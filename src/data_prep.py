import pandas as pd
import numpy as np
import yaml
import os
import sys
import argparse

def load_config(path="config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def load_and_clean(cfg):
    path = cfg['data_path']
    df = pd.read_json(path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Company','Date'])

    # keep relevant columns
    kpi = cfg['target_kpi']
    keep = ['Company','Date', kpi, 'OperatingExpenses','GrossProfit','Assets','StockholdersEquity']
    for col in keep:
        if col not in df.columns: df[col] = np.nan
    df = df[keep].replace('N/A', np.nan)

    # cast numeric
    for col in [kpi,'OperatingExpenses','GrossProfit','Assets','StockholdersEquity']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

def minimal_features(group, kpi):
    g = group.dropna(subset=[kpi]).copy()
    if len(g) < 3:
        return pd.DataFrame(columns=list(g.columns)+['lag_1','lag_2','diff_1','growth_1','mean_2','month','year'])
    g['lag_1'] = g[kpi].shift(1)
    g['lag_2'] = g[kpi].shift(2)
    g['diff_1'] = g[kpi] - g['lag_1']
    g['growth_1'] = (g[kpi] / g['lag_1'] - 1).replace([np.inf,-np.inf], np.nan)
    g['mean_2'] = (g[kpi] + g['lag_1']) / 2
    g['month'] = g['Date'].dt.month
    g['year'] = g['Date'].dt.year
    # we will predict next period from the last available row
    return g

def build_dataset(df, cfg):
    kpi = cfg['target_kpi']
    frames = []
    for _, g in df.groupby('Company'):
        m = minimal_features(g, kpi)
        if not m.empty:
            frames.append(m.tail(1))  # last row per company
    if len(frames) == 0:
        return pd.DataFrame()
    X = pd.concat(frames, axis=0)

    # naive & drift baselines (predict t+1 using last 2 points)
    X['arima_naive'] = X['lag_1']
    X['arima_drift'] = X['lag_1'] + (X['lag_1'] - X['lag_2'])

    return X

def save_artifacts(df, name, cfg):
    os.makedirs(cfg['artifacts_dir'], exist_ok=True)
    out = os.path.join(cfg['artifacts_dir'], name)
    df.to_parquet(out, index=False)
    print(f"Saved {out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data preparation for financial forecasting')
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    df = load_and_clean(cfg)
    X = build_dataset(df, cfg)
    save_artifacts(X, "prepared_last_rows.parquet", cfg)
