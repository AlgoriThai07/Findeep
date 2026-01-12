import pandas as pd
import numpy as np
import yaml
import os
import argparse


def load_config(path="config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_and_clean(cfg):
    path = cfg['data_path']
    df = pd.read_json(path)
    # normalize types
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.sort_values(['Company', 'Date'])

    kpi = cfg['target_kpi']
    keep = ['Company', 'Date', kpi, 'OperatingExpenses', 'GrossProfit', 'Assets', 'StockholdersEquity']
    for col in keep:
        if col not in df.columns:
            df[col] = np.nan
    df = df[keep].replace('N/A', np.nan)

    # cast numeric
    for col in [kpi, 'OperatingExpenses', 'GrossProfit', 'Assets', 'StockholdersEquity']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def add_features_per_group(g: pd.DataFrame, kpi: str) -> pd.DataFrame:
    g = g.copy()
    # require at least 3 to get lag_2 etc., but we keep rows even if NaNs; model will mask
    g['lag_1'] = g[kpi].shift(1)
    g['lag_2'] = g[kpi].shift(2)
    g['diff_1'] = g[kpi] - g['lag_1']
    g['growth_1'] = (g[kpi] / g['lag_1'] - 1).replace([np.inf, -np.inf], np.nan)
    g['mean_2'] = (g[kpi] + g['lag_1']) / 2
    g['month'] = g['Date'].dt.month
    g['year'] = g['Date'].dt.year
    # naive & drift baselines per row
    g['arima_naive'] = g['lag_1']
    g['arima_drift'] = g['lag_1'] + (g['lag_1'] - g['lag_2'])
    return g


def build_full_timeseries(df: pd.DataFrame, cfg):
    kpi = cfg['target_kpi']
    frames = []
    for _, g in df.groupby('Company'):
        m = add_features_per_group(g, kpi)
        if not m.empty:
            frames.append(m)
    if len(frames) == 0:
        return pd.DataFrame()
    X = pd.concat(frames, axis=0)
    return X


def save_artifacts(df, name, cfg):
    os.makedirs(cfg['artifacts_dir'], exist_ok=True)
    out = os.path.join(cfg['artifacts_dir'], name)
    df.to_parquet(out, index=False)
    print(f"Saved {out} (rows={len(df)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Full timeseries data prep (features for all rows)')
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = load_and_clean(cfg)
    X = build_full_timeseries(df, cfg)
    save_artifacts(X, "prepared_timeseries.parquet", cfg)
