"""Run baseline regression experiments on daily processed data."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

from utils import io_utils, metrics_utils, model_utils, plotting_utils, split_utils

FREQUENCY = "daily"
TASK = "regression"
TARGET_COLUMN = "target_regression_return"
RESULTS_ROOT = Path("data/results/experiments/daily/regression")
DROP_COLUMNS = ["date", "ticker"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline regression models on daily features")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to global config file")
    parser.add_argument("--dataset", type=str, help="Optional explicit path to processed dataset")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Training ratio for time split")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation ratio for time split")
    parser.add_argument("--ticker", type=str, help="Restrict to single ticker dataset")
    return parser.parse_args()


def load_dataset(args: argparse.Namespace, config: dict) -> pd.DataFrame:
    if args.dataset:
        path = Path(args.dataset)
        if path.suffix == ".csv":
            df = io_utils.load_csv(path)
        else:
            df = io_utils.load_parquet(path)
        return df
    processed_dir = Path(config["data"]["processed_daily_dir"])
    storage_format = config["data"].get("storage_format", "parquet")
    ticker = args.ticker.lower() if args.ticker else "all"
    candidate = processed_dir / f"{ticker}.{storage_format}"
    if not candidate.exists():
        raise FileNotFoundError(f"Processed dataset not found: {candidate}")
    if storage_format == "csv":
        return io_utils.load_csv(candidate)
    return io_utils.load_parquet(candidate)


def to_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [col for col in DROP_COLUMNS if col in df.columns]
    return df.drop(columns=drop_cols)


def create_run_dir(model_name: str) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = io_utils.path_for_run(RESULTS_ROOT, f"{timestamp}_{model_name}")
    return run_dir


def feature_names(frame: pd.DataFrame) -> Iterable[str]:
    return frame.columns


def maybe_plot_feature_importance(model, features: Iterable[str], output_dir: Path) -> None:
    estimator = model
    if hasattr(model, "named_steps"):
        estimator = model.named_steps.get("model", model)
    if hasattr(estimator, "feature_importances_"):
        plotting_utils.plot_feature_importance(
            estimator.feature_importances_,
            list(features),
            output_dir / "feature_importance.png",
        )


def run_experiment(args: argparse.Namespace, config: dict) -> None:
    df = load_dataset(args, config)
    df = df.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)

    split = split_utils.temporal_train_val_test_split(
        df,
        target_cols=[TARGET_COLUMN],
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    X_train = to_feature_matrix(split.X_train)
    X_val = to_feature_matrix(split.X_val)
    X_test = to_feature_matrix(split.X_test)

    y_train = split.y_train[TARGET_COLUMN]
    y_val = split.y_val[TARGET_COLUMN]
    y_test = split.y_test[TARGET_COLUMN]

    models = config["modeling"]["experiment_models"][TASK]

    for spec in models:
        model_name = spec["model"]
        params = spec.get("params", {})
        run_dir = create_run_dir(model_name)
        estimator = model_utils.fit_model(X_train, y_train, TASK, model_name, params)

        val_pred, _ = model_utils.predict(estimator, X_val)
        test_pred, _ = model_utils.predict(estimator, X_test)

        val_metrics = metrics_utils.regression_metrics(y_val, val_pred)
        test_metrics = metrics_utils.regression_metrics(y_test, test_pred)

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
        io_utils.save_csv(preds_df, run_dir / "predictions.csv")

        plotting_utils.plot_actual_vs_predicted(y_test, test_pred, run_dir / "actual_vs_pred.png")
        plotting_utils.plot_residuals(y_test, test_pred, run_dir / "residuals.png")
        maybe_plot_feature_importance(estimator, feature_names(X_train), run_dir)

        io_utils.save_model(estimator, run_dir / "model.pkl")
        print(f"Completed experiment -> {run_dir}")


def main() -> None:
    args = parse_args()
    config = io_utils.load_yaml(args.config)
    run_experiment(args, config)


if __name__ == "__main__":
    main()

