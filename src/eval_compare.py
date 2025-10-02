import os, yaml, pandas as pd, numpy as np, sys, argparse

def load_config(path="config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def metrics_table(df, kpi):
    rows = []
    for _, r in df.iterrows():
        y = r.get(kpi, np.nan)
        preds = {
            'naive': r.get('arima_naive', np.nan),
            'drift': r.get('arima_drift', np.nan),
            'xgb': r.get('xgb_point', np.nan)
        }
        for name, yhat in preds.items():
            if not (np.isnan(y) or np.isnan(yhat)):
                ae = abs(y - yhat); se = (y - yhat)**2
            else:
                ae = np.nan; se = np.nan
            rows.append({'Company': r.get('Company','UNK'), 'model': name, 'AE': ae, 'SE': se})
    mt = pd.DataFrame(rows)
    if mt.empty:
        return mt, {'MAE': np.nan, 'RMSE': np.nan}
    mtg = mt.groupby('model').agg(MAE=('AE','mean'), RMSE=('SE', lambda s: np.sqrt(s.mean()))).reset_index()
    return mtg, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model performance')
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    pred_path = os.path.join(cfg['artifacts_dir'], 'predictions.parquet')
    df = pd.read_parquet(pred_path)
    table, _ = metrics_table(df, cfg['target_kpi'])
    out_csv = os.path.join(cfg['artifacts_dir'], 'metrics.csv')
    table.to_csv(out_csv, index=False)
    print("Saved metrics:", out_csv)
    print(table)
