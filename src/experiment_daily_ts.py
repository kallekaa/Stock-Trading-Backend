"""Run baseline time-series forecasting experiments on daily data."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from utils import io_utils, metrics_utils, model_utils, plotting_utils, split_utils

FREQUENCY = "daily"
TASK = "time_series"
RESULTS_ROOT = Path("data/results/experiments/daily/time_series")
DROP_COLUMNS = ["date", "ticker"]
TARGET_TEMPLATE = "target_forecast_close_t+{horizon}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline time-series models on daily data")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--dataset", type=str, help="Optional explicit dataset path")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--ticker", type=str, help="Restrict to single ticker")
    parser.add_argument("--horizons", type=int, nargs="*", help="Override forecasting horizons")
    return parser.parse_args()


def load_dataset(args: argparse.Namespace, config: dict) -> pd.DataFrame:
    if args.dataset:
        path = Path(args.dataset)
        return io_utils.load_csv(path) if path.suffix == ".csv" else io_utils.load_parquet(path)
    processed_dir = Path(config["data"]["processed_daily_dir"])
    storage_format = config["data"].get("storage_format", "parquet")
    ticker = args.ticker.lower() if args.ticker else "all"
    candidate = processed_dir / f"{ticker}.{storage_format}"
    if not candidate.exists():
        raise FileNotFoundError(candidate)
    return io_utils.load_csv(candidate) if storage_format == "csv" else io_utils.load_parquet(candidate)


def to_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [col for col in DROP_COLUMNS if col in df.columns]
    return df.drop(columns=drop_cols)


def run_experiment(args: argparse.Namespace, config: dict) -> None:
    df = load_dataset(args, config)
    horizons = args.horizons or config["modeling"]["horizons"]["forecasting"]
    models = config["modeling"]["experiment_models"][TASK]

    for horizon in horizons:
        target_col = TARGET_TEMPLATE.format(horizon=horizon)
        if target_col not in df.columns:
            print(f"Skipping horizon {horizon}: missing target column {target_col}")
            continue
        target_df = df.dropna(subset=[target_col]).reset_index(drop=True)
        if target_df.empty:
            print(f"Skipping horizon {horizon}: no data after dropping null targets")
            continue
        split = split_utils.temporal_train_val_test_split(
            target_df,
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

        for spec in models:
            model_name = spec["model"]
            params = spec.get("params", {})
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            run_dir = io_utils.path_for_run(
                RESULTS_ROOT,
                f"h{horizon}",
                f"{timestamp}_{model_name}",
            )
            estimator = model_utils.fit_model(X_train, y_train, TASK, model_name, params)
            val_pred, _ = model_utils.predict(estimator, X_val)
            test_pred, _ = model_utils.predict(estimator, X_test)

            val_metrics = metrics_utils.forecasting_metrics(y_val, val_pred)
            test_metrics = metrics_utils.forecasting_metrics(y_test, test_pred)

            io_utils.save_yaml(
                {
                    "task": TASK,
                    "frequency": FREQUENCY,
                    "horizon": horizon,
                    "model": model_name,
                    "params": params,
                    "train_ratio": args.train_ratio,
                    "val_ratio": args.val_ratio,
                },
                run_dir / "params.yaml",
            )
            io_utils.save_json({"val": val_metrics, "test": test_metrics}, run_dir / "metrics.json")

            preds_df = pd.DataFrame(
                {
                    "date": split.X_test["date"].values,
                    "ticker": split.X_test.get("ticker", pd.Series(index=split.X_test.index, dtype=str)),
                    "y_true": y_test.values,
                    "y_pred": test_pred,
                }
            )
            io_utils.save_csv(preds_df, run_dir / "predictions.csv")

            plotting_utils.plot_actual_vs_predicted(y_test, test_pred, run_dir / "actual_vs_pred.png")
            plotting_utils.plot_residuals(y_test, test_pred, run_dir / "residuals.png")

            io_utils.save_model(estimator, run_dir / "model.pkl")
            print(f"Completed horizon {horizon} -> {run_dir}")


def main() -> None:
    args = parse_args()
    config = io_utils.load_yaml(args.config)
    run_experiment(args, config)


if __name__ == "__main__":
    main()

