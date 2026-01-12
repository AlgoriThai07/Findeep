"""
plot_backtest.py
Slide-ready bar chart comparing actual vs predicted revenue.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_backtest_bar(comparison_df: pd.DataFrame, out_png: str):
    """
    Create a side-by-side bar chart per Company: actual vs xgb_point.
    
    Args:
        comparison_df: DataFrame with ['Company', 'actual', 'xgb_point']
        out_png: Output PNG file path
    """
    if comparison_df.empty:
        print("No data to plot.")
        return
    
    # Convert to billions
    df = comparison_df.copy()
    df['actual_B'] = df['actual'] / 1e9
    df['xgb_point_B'] = df['xgb_point'] / 1e9
    
    companies = df['Company'].unique()
    x = np.arange(len(companies))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    actual_vals = [df[df['Company'] == c]['actual_B'].values[0] for c in companies]
    pred_vals = [df[df['Company'] == c]['xgb_point_B'].values[0] for c in companies]
    
    bars1 = ax.bar(x - width/2, actual_vals, width, label='Actual', color='steelblue')
    bars2 = ax.bar(x + width/2, pred_vals, width, label='XGB Prediction', color='coral')
    
    ax.set_xlabel('Company', fontsize=12)
    ax.set_ylabel('Revenue (Billions $)', fontsize=12)
    ax.set_title('Backtest: Actual vs XGBoost Predicted Revenue', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(companies, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"Saved plot: {out_png}")
    plt.close()
