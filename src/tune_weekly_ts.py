"""Hyperparameter tuning and final model selection for weekly time-series forecasting."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from utils import io_utils, metrics_utils, model_utils, split_utils

FREQUENCY = "weekly"
TASK = "time_series"
TARGET_TEMPLATE = "target_forecast_close_t+{horizon}"
TUNING_ROOT = Path("data/results/tuning/weekly/time_series")
FINAL_MODEL_DIR = Path("data/results/final_models/weekly/time_series")
DROP_COLUMNS = ["date", "ticker"]


SEARCH_SPACE: Dict[str, List[Dict[str, Any]]] = {
    "RandomForestRegressor": [
        {"n_estimators": 200, "max_depth": 6, "random_state": 42},
        {"n_estimators": 400, "max_depth": 8, "random_state": 42},
        {"n_estimators": 600, "max_depth": None, "random_state": 42},
    ],
    "GradientBoostingRegressor": [
        {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3, "random_state": 42},
        {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 4, "random_state": 42},
        {"n_estimators": 400, "learning_rate": 0.03, "max_depth": 3, "random_state": 42},
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune weekly forecasting models")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--dataset", type=str, help="Optional explicit dataset path")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--ticker", type=str, help="Restrict to single ticker dataset")
    parser.add_argument("--cv-splits", type=int, help="Override TimeSeriesSplit splits")
    parser.add_argument("--horizon", type=int, help="Forecast horizon to tune")
    return parser.parse_args()


def load_dataset(args: argparse.Namespace, config: dict) -> pd.DataFrame:
    if args.dataset:
        path = Path(args.dataset)
        return io_utils.load_csv(path) if path.suffix == ".csv" else io_utils.load_parquet(path)
    processed_dir = Path(config["data"]["processed_weekly_dir"])
    storage_format = config["data"].get("storage_format", "parquet")
    ticker = args.ticker.lower() if args.ticker else "all"
    candidate = processed_dir / f"{ticker}.{storage_format}"
    if not candidate.exists():
        raise FileNotFoundError(candidate)
    return io_utils.load_csv(candidate) if storage_format == "csv" else io_utils.load_parquet(candidate)


def to_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [col for col in DROP_COLUMNS if col in df.columns]
    return df.drop(columns=drop_cols)


def evaluate_param_set(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    params: Dict[str, Any],
    n_splits: int,
) -> Tuple[float, List[Dict[str, float]]]:
    splitter = TimeSeriesSplit(n_splits=n_splits)
    scores: List[float] = []
    fold_metrics: List[Dict[str, float]] = []
    for train_idx, val_idx in splitter.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        estimator = model_utils.fit_model(X_train, y_train, TASK, model_name, params)
        preds, _ = model_utils.predict(estimator, X_val)
        metrics = metrics_utils.forecasting_metrics(y_val, preds)
        fold_metrics.append(metrics)
        scores.append(-metrics["rmse"])
    return float(np.mean(scores)), fold_metrics


def main() -> None:
    args = parse_args()
    config = io_utils.load_yaml(args.config)
    df = load_dataset(args, config)

    horizons = config["modeling"]["horizons"]["forecasting"]
    horizon = args.horizon or (horizons[0] if horizons else 1)
    target_col = TARGET_TEMPLATE.format(horizon=horizon)

    if target_col not in df.columns:
        raise KeyError(f"Missing target column {target_col} for horizon {horizon}")

    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    split = split_utils.temporal_train_val_test_split(
        df,
        target_cols=[target_col],
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    X_train = to_feature_matrix(split.X_train)
    X_val = to_feature_matrix(split.X_val)
    X_test = to_feature_matrix(split.X_test)

    y_train = split.y_train[target_col]
    y_val = split.y_val[target_col]
    y_test = split.y_test[target_col]

    trainval_features = pd.concat([X_train, X_val], axis=0)
    trainval_target = pd.concat([y_train, y_val], axis=0)

    n_splits = args.cv_splits or config["modeling"]["tuning"]["cv_splits"]

    records = []
    best_score = -np.inf
    best_spec: Dict[str, Any] | None = None

    for model_name, param_list in SEARCH_SPACE.items():
        for params in param_list:
            avg_score, fold_metrics = evaluate_param_set(trainval_features, trainval_target, model_name, params, n_splits)
            record = {
                "model": model_name,
                "params": json.dumps(params),
                "avg_score": avg_score,
                "fold_metrics": json.dumps(fold_metrics),
            }
            records.append(record)
            if avg_score > best_score:
                best_score = avg_score
                best_spec = {
                    "model": model_name,
                    "params": params,
                    "fold_metrics": fold_metrics,
                }
            print(f"Scored model={model_name} params={params} avg_score={avg_score:.4f}")

    if not best_spec:
        raise RuntimeError("No valid model configuration found during tuning")

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    tuning_run_dir = io_utils.path_for_run(TUNING_ROOT, f"h{horizon}", f"{timestamp}_{best_spec['model']}")

    cv_df = pd.DataFrame.from_records(records)
    io_utils.save_csv(cv_df, tuning_run_dir / "cv_results.csv")
    io_utils.save_yaml(
        {
            "best_model": best_spec["model"],
            "best_params": best_spec["params"],
            "cv_metric": "-rmse (higher is better)",
            "avg_score": best_score,
            "horizon": horizon,
        },
        tuning_run_dir / "best_summary.yaml",
    )

    best_estimator = model_utils.fit_model(trainval_features, trainval_target, TASK, best_spec["model"], best_spec["params"])
    test_pred, _ = model_utils.predict(best_estimator, X_test)
    test_metrics = metrics_utils.forecasting_metrics(y_test, test_pred)

    final_dir = io_utils.ensure_directory(FINAL_MODEL_DIR)
    io_utils.save_model(best_estimator, final_dir / "model.pkl")
    io_utils.save_yaml(
        {
            "task": TASK,
            "frequency": FREQUENCY,
            "model": best_spec["model"],
            "params": best_spec["params"],
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "cv_splits": n_splits,
            "horizon": horizon,
        },
        final_dir / "params.yaml",
    )
    io_utils.save_json({"test": test_metrics}, final_dir / "metrics_test.json")

    notes_path = final_dir / "notes.md"
    notes_path.write_text(
        "\n".join(
            [
                f"# Weekly Time Series Final Model ({timestamp})",
                f"Horizon: {horizon}",
                f"Model: {best_spec['model']}",
                f"Parameters: {best_spec['params']}",
                f"Average tuning score (-rmse): {best_score:.4f}",
                f"Test metrics: {json.dumps(test_metrics, indent=2)}",
                "",
                "Tuning details saved under data/results/tuning/weekly/time_series/",
            ]
        ),
        encoding="utf-8",
    )

    io_utils.save_json({"test": test_metrics}, tuning_run_dir / "test_metrics.json")
    print(f"Saved weekly time-series model -> {final_dir}")


if __name__ == "__main__":
    main()

