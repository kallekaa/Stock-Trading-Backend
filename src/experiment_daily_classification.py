"""Run baseline classification experiments on daily processed data."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from utils import io_utils, metrics_utils, model_utils, plotting_utils, split_utils

FREQUENCY = "daily"
TASK = "classification"
TARGET_COLUMN = "target_classification_direction"
RESULTS_ROOT = Path("data/results/experiments/daily/classification")
DROP_COLUMNS = ["date", "ticker"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline classification models on daily features")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--dataset", type=str, help="Optional explicit dataset path")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--ticker", type=str, help="Restrict to single ticker")
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


def create_run_dir(model_name: str) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return io_utils.path_for_run(RESULTS_ROOT, f"{timestamp}_{model_name}")


def run_experiment(args: argparse.Namespace, config: dict) -> None:
    df = load_dataset(args, config)
    df = df.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    split = split_utils.temporal_train_val_test_split(
        df,
        target_cols=[TARGET_COLUMN],
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    X_train = to_feature_matrix(split.X_train)
    X_val = to_feature_matrix(split.X_val)
    X_test = to_feature_matrix(split.X_test)

    y_train = split.y_train[TARGET_COLUMN].astype(int)
    y_val = split.y_val[TARGET_COLUMN].astype(int)
    y_test = split.y_test[TARGET_COLUMN].astype(int)

    models = config["modeling"]["experiment_models"][TASK]

    for spec in models:
        model_name = spec["model"]
        params = spec.get("params", {})
        run_dir = create_run_dir(model_name)
        estimator = model_utils.fit_model(X_train, y_train, TASK, model_name, params)

        val_pred, val_prob = model_utils.predict(estimator, X_val)
        test_pred, test_prob = model_utils.predict(estimator, X_test)

        val_metrics = metrics_utils.classification_metrics(y_val, val_pred, val_prob)
        test_metrics = metrics_utils.classification_metrics(y_test, test_pred, test_prob)

        io_utils.save_yaml(
            {
                "task": TASK,
                "frequency": FREQUENCY,
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
        if test_prob is not None:
            prob_df = pd.DataFrame(test_prob, columns=[f"prob_class_{i}" for i in range(test_prob.shape[1])])
            preds_df = pd.concat([preds_df.reset_index(drop=True), prob_df], axis=1)
        io_utils.save_csv(preds_df, run_dir / "predictions.csv")

        plotting_utils.plot_confusion_matrix(y_test, test_pred, run_dir / "confusion_matrix.png")
        if test_prob is not None:
            plotting_utils.plot_roc_curve(y_test, np.asarray(test_prob), run_dir / "roc_curve.png")

        io_utils.save_model(estimator, run_dir / "model.pkl")
        print(f"Completed classification experiment -> {run_dir}")


def main() -> None:
    args = parse_args()
    config = io_utils.load_yaml(args.config)
    run_experiment(args, config)


if __name__ == "__main__":
    main()

