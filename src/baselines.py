import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import os, yaml, sys, argparse

FEATS = ["lag_1","lag_2","diff_1","growth_1","mean_2","month","year",
         "OperatingExpenses","GrossProfit","Assets","StockholdersEquity"]

def load_config(path="config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def ewma_vol(residuals, lam=0.94):
    var = 0.0
    for r in residuals:
        var = lam*var + (1-lam)*(r**2)
    return np.sqrt(var + 1e-12)

def train_xgb(df, cfg):
    kpi = cfg['target_kpi']
    dfx = df.dropna(subset=FEATS+[kpi]).copy()
    if dfx.empty:
        raise ValueError("No rows available to train XGBoost. Check data and features.")
    
    print(f"Training XGBoost with {len(dfx)} samples after filtering")
    X = dfx[FEATS].astype(float).values
    y = dfx[kpi].astype(float).values  # target = next period level (pooled)

    # Handle small datasets - if we have fewer than 5 samples, skip validation split
    if len(dfx) < 5:
        print(f"Small dataset ({len(dfx)} samples) - training without validation split")
        model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1,
                                 subsample=0.9, colsample_bytree=0.9, tree_method="hist", random_state=cfg['seed'])
        model.fit(X, y, verbose=False)
        return model
    else:
        # Normal train/validation split for larger datasets
        Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=cfg['validation_fraction'], random_state=cfg['seed'])
        model = xgb.XGBRegressor(n_estimators=600, max_depth=4, learning_rate=0.05,
                                 subsample=0.9, colsample_bytree=0.9, tree_method="hist", random_state=cfg['seed'])
        model.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
        return model

def predict_all(df, model, cfg):
    kpi = cfg['target_kpi']
    out = df.copy()
    mask = out[FEATS].notnull().all(axis=1)
    out.loc[mask, 'xgb_point'] = model.predict(out.loc[mask, FEATS].astype(float))
    # EWMA interval width from pooled residuals (proxy for GARCH)
    valid = out.dropna(subset=[kpi,'xgb_point'])
    resid = valid[kpi] - valid['xgb_point']
    sigma = ewma_vol(resid.values) if len(valid) > 0 else 0.0
    out['pi_lower'] = out['xgb_point'] - 1.96*sigma
    out['pi_upper'] = out['xgb_point'] + 1.96*sigma
    return out

def save_predictions(df, cfg):
    os.makedirs(cfg['artifacts_dir'], exist_ok=True)
    path = os.path.join(cfg['artifacts_dir'], "predictions.parquet")
    df.to_parquet(path, index=False)
    print(f"Saved {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train baseline models')
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    prep_path = os.path.join(cfg['artifacts_dir'], "prepared_last_rows.parquet")
    df = pd.read_parquet(prep_path)
    model = train_xgb(df, cfg)
    pred = predict_all(df, model, cfg)
    save_predictions(pred, cfg)
