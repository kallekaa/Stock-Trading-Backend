# Stock Market ML: Daily & Weekly Forecasting Framework

This repository provides a modular, end-to-end machine learning stack for predicting stock market movements at daily and weekly resolutions. Each pipeline stage is implemented as a standalone script so that data scientists can iterate, debug, and deploy individual components without blocking on the rest of the system.

## Features
- Handles three core problem types: time-series forecasting, regression on future returns, and multi-class direction classification.
- Supports both daily quotes and weekly aggregates with shared utilities for IO, feature engineering, splitting, metrics, plotting, and model management.
- Stores intermediate artifacts on disk with a consistent hierarchy under `data/` so experiments remain reproducible without an external tracking server.
- Ships with baseline models, tuning workflows, and prediction scripts that you can extend with new assets or algorithms.

## Quick Start
1. **Create environment and install dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```
2. **Populate global configuration**
   Edit `config.yaml` to list tickers, date ranges, and feature settings. Defaults cover AAPL, MSFT, and SPY.
3. **Pull raw data**
   ```bash
   python -m src.ingest_data --config config.yaml
   ```
4. **Process features**
   ```bash
   python -m src.process_daily --config config.yaml
   python -m src.process_weekly --config config.yaml
   ```

## Pipeline Stages

### 1. Ingestion (`src/ingest_data.py`)
Downloads OHLCV data via yfinance (or reads a provided CSV) and writes normalized snapshots into `data/raw/daily_quotes/`. Column names are standardized to `date`, `ticker`, `open`, `high`, `low`, `close`, `adj_close`, and `volume`.

### 2. Processing (`src/process_daily.py`, `src/process_weekly.py`)
Transforms raw quotes into feature-rich datasets, computes targets for all task types, and saves the outputs under `data/processed/daily/` and `data/processed/weekly/`. Feature engineering helpers live in `src/utils/feature_utils.py`.

### 3. Experimentation (`src/experiment_*`)
Runs baseline models for each (frequency, task) combination. Each script loads processed data, performs chronological train/val/test splits, trains the configured models, and persists metrics, plots, and serialized estimators to `data/results/experiments/...`.

### 4. Tuning & Final Models (`src/tune_*`)
Manual time-series cross-validation loops search the parameter space for the best performing configuration. Final estimators, params, and held-out metrics are stored under `data/results/final_models/<frequency>/<task>/`, while CV details land in `data/results/tuning/...`.

### 5. Inference (`src/predict_*`)
Loads the finalized models, scores the latest processed features, and writes timestamped prediction files to `data/predictions/<frequency>/<task>/`. Metadata JSON files capture the model configuration, horizon, and output path for traceability.

## Utilities
- `src/utils/io_utils.py`: serialization helpers (CSV, Parquet, JSON, YAML, joblib) and run directory utilities.
- `src/utils/feature_utils.py`: rolling statistics, momentum indicators, RSI/MACD, and target constructors.
- `src/utils/split_utils.py`: chronological train/val/test splits plus walk-forward iterators.
- `src/utils/metrics_utils.py`: regression, classification, and forecasting metrics wrappers.
- `src/utils/model_utils.py`: model factory wrapping scikit-learn estimators with optional scaling.
- `src/utils/plotting_utils.py`: diagnostic plots for experiments and tuning.

## Directory Layout
```
+-- config.yaml
+-- data/
¦   +-- raw/
¦   +-- processed/
¦   +-- results/
¦   +-- predictions/
+-- src/
¦   +-- ingest, process, experiment, tune, and predict scripts
¦   +-- utils/
¦   +-- config_schemas/
+-- requirements.txt
```

## Next Steps
- Extend `config.yaml` with additional tickers or feature sets.
- Plug in alternative models by updating `model_utils.MODEL_REGISTRY` and the experiment/tuning search spaces.
- Automate scheduled ingestion and inference once the baseline pipeline is validated.

