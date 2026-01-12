# Findeep

Findeep is a robust **Financial Forecasting Pipeline** that predicts next-quarter corporate revenues using fundamental financial data from SEC filings (10-K and 10-Q).

It features a strict **Backtesting Framework** designed to simulate real-world trading/forecasting conditions by preventing lookahead bias and data leakage.

## Key Features

- **Automated Data Ingestion**: Custom scraping pipelines for SEC EDGAR that fetch, parse, and structure XBRL/XML data from 10-Q (Quarterly) and 10-K (Annual) reports.
- **Leakage-Free Backtest Engine**: A specialized validation framework (`run_proper_backtest.py`) that strictly segregates training and testing data by time. It ensures that predictions for Quarter $T$ use *only* information available before $T$.
- **Advanced Feature Engineering**: Generates time-series features such as quarter-over-quarter growth rates, lagged indicators ($t-1, t-2$), and rolling means.
- **Machine Learning**: Uses **XGBoost** (Gradient Boosting) for regression forecasting, tuned for small-sample financial time-series.

## Repository Structure

- `run_proper_backtest.py`: **Main Entry Point**. Orchestrates the entire backtesting workflow (Load Data -> Train -> Predict -> Evaluate).
- `tesla_10q_pipeline.py`: Data pipeline for Tesla (TSLA). Fetches 10-Q/10-K filings and extracts financial tags (Revenue, Assets, etc.).
- `JP_10q_pipeline_full.py`: Data pipeline for JP Morgan (JPM).
- `filter_and_combine_excel.py`: Utility to merge individual company datasets into a unified format for training.
- `src/`: Core logic library.
  - `backtest_utils.py`: Contains the `assert_no_leakage` logic and test set construction.
  - `supervised_prep.py`: Feature engineering logic (lag generation).
  - `train_next_quarter.py`: XGBoost model training wrapper.

## Getting Started

### 1. Prerequisites

Ensure you have Python 3.10+ installed.

```bash
pip install -r requirements.txt
```

### 2. Run Data Pipelines

Fetch the latest financial data directly from the SEC:

```bash
# Fetch Tesla Data (10-Q & 10-K)
python tesla_10q_pipeline.py

# Fetch JP Morgan Data (10-Q & 10-K)
python JP_10q_pipeline_full.py
```

Combine the raw excel files into a training dataset:

```bash
python filter_and_combine_excel.py
```

### 3. Run the Backtest

Execute the full forecasting pipeline:

```bash
python run_proper_backtest.py
```

This will:
1.  Load the combined data.
2.  Train an XGBoost model on historical data (e.g., up to 2024-03-31).
3.  Predict the specific target quarter (e.g., 2024-06-30).
4.  Save results to `artifacts/backtest_predictions.parquet`.
5.  Print evaluation metrics (MAE, RMSE).

## Methodological Note: "Proper" Backtesting

Unlike standard cross-validation, financial time-series require strict temporal ordering. This project implements a **Walk-Forward Validation** approach.

- **Training**: Uses all available history up to `train_end_date`.
- **Testing**: Predicts exactly one step ahead (`test_date`).
- **Safety**: The `src/backtest_utils.py` module enforces that feature generation for the test set *never* accesses future rows, preventing the common "lookahead bias" error in financial ML.
