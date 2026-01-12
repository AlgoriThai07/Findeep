# run_xgb_baseline.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import os, yaml, argparse

FEATS = [
    "lag_1","lag_2","diff_1","growth_1","mean_2","month","year",
    "OperatingExpenses","GrossProfit","Assets","StockholdersEquity"
]

def load_config(path="config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def ewma_vol(residuals, lam=0.94):
    var = 0.0
    for r in residuals:
        var = lam*var + (1-lam)*(r**2)
    return float(np.sqrt(var + 1e-12))

def ensure_year_quarter(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Prefer Date if present
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        if df["Date"].notna().any():
            df["year"] = df["Date"].dt.year.astype("Int64")
            df["quarter"] = ((df["Date"].dt.month - 1)//3 + 1).astype("Int64")
    # Fallback: derive quarter from month if missing
    if "quarter" not in df.columns or df["quarter"].isna().all():
        if "month" in df.columns:
            df["quarter"] = ((df["month"].astype(int) - 1)//3 + 1).astype("Int64")
    # If year column missing but Date gave us year, keep it; else leave existing year
    if "year" not in df.columns:
        if "Date" in df.columns and df["Date"].notna().any():
            df["year"] = df["Date"].dt.year.astype("Int64")
    return df

def train_xgb(df: pd.DataFrame, cfg):
    kpi = cfg['target_kpi']
    # Make sure features exist; if some are missing, drop them from FEATS for this run
    present_feats = [c for c in FEATS if c in df.columns]
    if len(present_feats) == 0:
        raise ValueError("No required features present. Check your prepared dataset columns.")

    # Filter rows with target and all used features
    use_cols = present_feats + [kpi]
    dfx = df.dropna(subset=use_cols).copy()
    if dfx.empty:
        raise ValueError(f"No rows available to train XGBoost for target '{kpi}'. Check data and features.")

    print(f"Training XGBoost on KPI='{kpi}' with {len(dfx)} samples and {len(present_feats)} features")
    X = dfx[present_feats].astype(float).values
    y = dfx[kpi].astype(float).values

    # Small dataset fallback
    if len(dfx) < max(5, len(present_feats) + 1):
        print(f"Small dataset ({len(dfx)} samples) — training without validation split")
        model = xgb.XGBRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9, tree_method="hist",
            random_state=cfg.get('seed', 42)
        )
        model.fit(X, y, verbose=False)
        return model, present_feats
    else:
        Xtr, Xval, ytr, yval = train_test_split(
            X, y, test_size=cfg.get('validation_fraction', 0.2),
            random_state=cfg.get('seed', 42)
        )
        model = xgb.XGBRegressor(
            n_estimators=600, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, tree_method="hist",
            random_state=cfg.get('seed', 42)
        )
        model.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
        return model, present_feats

def predict_all(df: pd.DataFrame, model, used_feats, cfg):
    kpi = cfg['target_kpi']
    out = df.copy()

    # Predict where all used features are available
    mask = out[used_feats].notnull().all(axis=1)
    if mask.any():
        out.loc[mask, 'xgb_point'] = model.predict(out.loc[mask, used_feats].astype(float))
    else:
        out['xgb_point'] = np.nan

    # EWMA interval width from residuals (pooled) — proxy for GARCH
    valid = out.dropna(subset=[kpi, 'xgb_point'])
    if len(valid) > 0:
        resid = valid[kpi].astype(float) - valid['xgb_point'].astype(float)
        sigma = ewma_vol(resid.values)
    else:
        sigma = 0.0
    out['pi_lower'] = out['xgb_point'] - 1.96*sigma
    out['pi_upper'] = out['xgb_point'] + 1.96*sigma

    return out

def finalize_columns(df: pd.DataFrame, cfg):
    """Ensure columns required by finetune_2023 & inference exist."""
    df = ensure_year_quarter(df)

    # Ensure Company and KPI exist
    if "Company" not in df.columns:
        df["Company"] = "Unknown"
    if "KPI" not in df.columns:
        df["KPI"] = cfg['target_kpi']

    # Keep common baseline columns if present (ARIMA, etc.)
    for col in ["arima_naive", "arima_drift"]:
        if col not in df.columns:
            df[col] = np.nan

    # Order is not critical, but nice for readability
    preferred = [
        "Company", "KPI", "Date", "year", "quarter", "month",
        cfg['target_kpi'], "xgb_point", "pi_lower", "pi_upper",
        "arima_naive", "arima_drift",
        "OperatingExpenses","GrossProfit","Assets","StockholdersEquity",
        "lag_1","lag_2","diff_1","growth_1","mean_2"
    ]
    # Put preferred first, then everything else
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    return df[cols]

def save_predictions(df: pd.DataFrame, cfg):
    os.makedirs(cfg['artifacts_dir'], exist_ok=True)
    # Primary 2023-specific output
    path_2023 = os.path.join(cfg['artifacts_dir'], "predictions_2023.parquet")
    df.to_parquet(path_2023, index=False)
    print(f"Saved {path_2023} (rows={len(df)})")

    # Also write legacy filename for compatibility with existing scripts
    legacy_path = os.path.join(cfg['artifacts_dir'], "predictions.parquet")
    try:
        df.to_parquet(legacy_path, index=False)
        print(f"Also wrote legacy {legacy_path} for compatibility")
    except Exception as e:
        print(f"Warning: failed to write legacy predictions.parquet: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train XGBoost baseline and write predictions.parquet')
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    args = parser.parse_args()

    cfg = load_config(args.config)
    # Prefer full timeseries if available
    prep_full = os.path.join(cfg['artifacts_dir'], "prepared_timeseries.parquet")
    prep_last = os.path.join(cfg['artifacts_dir'], "prepared_last_rows.parquet")
    prep_path = prep_full if os.path.exists(prep_full) else prep_last
    if not os.path.exists(prep_path):
        raise FileNotFoundError(
            f"Missing {prep_full} and {prep_last}. Run src/data_prep_full.py or src/data_prep.py first.")

    print(f"Using input: {prep_path}")
    df = pd.read_parquet(prep_path)
    df = ensure_year_quarter(df)

    model, used_feats = train_xgb(df, cfg)
    pred = predict_all(df, model, used_feats, cfg)
    pred = finalize_columns(pred, cfg)
    save_predictions(pred, cfg)
