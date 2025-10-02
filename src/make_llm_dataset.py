import os, json, yaml, pandas as pd, numpy as np

def load_config(path="config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def df_to_llm_rows(df, cfg):
    kpi = cfg['target_kpi']
    rows = []
    for _, r in df.iterrows():
        history = []
        # We only have the *last* row here; store lag_1 (t), lag_2 (t-1) as history
        if not np.isnan(r.get('lag_2', np.nan)):
            history.append({"t": -2, "value": float(r['lag_2'])})
        if not np.isnan(r.get('lag_1', np.nan)):
            history.append({"t": -1, "value": float(r['lag_1'])})

        prompt = {
            "role": "system",
            "content": "You are a financial forecasting assistant. Combine baseline predictions and tiny history to forecast the next period and explain briefly."
        }
        user = {
            "role": "user",
            "content": {
                "company": r.get('Company','UNKNOWN'),
                "kpi": kpi,
                "history": history,
                "baselines": {
                    "arima_naive": None if pd.isna(r.get('arima_naive')) else float(r['arima_naive']),
                    "arima_drift": None if pd.isna(r.get('arima_drift')) else float(r['arima_drift']),
                    "xgb_point": None if pd.isna(r.get('xgb_point')) else float(r['xgb_point'])
                },
                "exog": {
                    "OperatingExpenses": None if pd.isna(r.get('OperatingExpenses')) else float(r['OperatingExpenses']),
                    "GrossProfit": None if pd.isna(r.get('GrossProfit')) else float(r['GrossProfit']),
                    "Assets": None if pd.isna(r.get('Assets')) else float(r['Assets']),
                    "StockholdersEquity": None if pd.isna(r.get('StockholdersEquity')) else float(r['StockholdersEquity'])
                }
            }
        }
        # If true next value exists in the row, use it as supervised target; otherwise set None
        target_val = None if pd.isna(r.get(kpi)) else float(r[kpi])
        assistant = {
            "role": "assistant",
            "content": {
                "final_forecast": target_val,  # In production you may use xgb_point or midpoint as weak label if real y_{t+1} not present
                "explanation": "Choosing a conservative blend of drift and XGB given rising OPEX and mixed gross profit."
            }
        }
        rows.append({"messages":[prompt, user, assistant]})
    return rows

if __name__ == "__main__":
    cfg = load_config()
    pred_path = os.path.join(cfg['artifacts_dir'], "predictions.parquet")
    df = pd.read_parquet(pred_path)
    rows = df_to_llm_rows(df, cfg)
    os.makedirs(cfg['artifacts_dir'], exist_ok=True)
    # simple split
    n = len(rows)
    n_val = max(1, int(n * cfg['validation_fraction']))
    val = rows[:n_val]; tr = rows[n_val:]
    with open(os.path.join(cfg['artifacts_dir'], 'llm_train.jsonl'), 'w') as f:
        for r in tr: f.write(json.dumps(r)+"\n")
    with open(os.path.join(cfg['artifacts_dir'], 'llm_val.jsonl'), 'w') as f:
        for r in val: f.write(json.dumps(r)+"\n")
    print(f"Wrote {len(tr)} train and {len(val)} val rows.")
