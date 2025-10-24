"""Hyperparameter tuning and final model selection for weekly classification."""
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
TASK = "classification"
TARGET_COLUMN = "target_classification_direction"
TUNING_ROOT = Path("data/results/tuning/weekly/classification")
FINAL_MODEL_DIR = Path("data/results/final_models/weekly/classification")
DROP_COLUMNS = ["date", "ticker"]


SEARCH_SPACE: Dict[str, List[Dict[str, Any]]] = {
    "LogisticRegression": [
        {"max_iter": 1000, "C": 0.5, "class_weight": "balanced", "multi_class": "auto"},
        {"max_iter": 1000, "C": 1.0, "class_weight": "balanced", "multi_class": "auto"},
        {"max_iter": 1000, "C": 2.0, "class_weight": "balanced", "multi_class": "auto"},
    ],
    "RandomForestClassifier": [
        {"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 2, "class_weight": "balanced", "random_state": 42},
        {"n_estimators": 300, "max_depth": 8, "min_samples_leaf": 1, "class_weight": "balanced_subsample", "random_state": 42},
        {"n_estimators": 400, "max_depth": None, "min_samples_leaf": 2, "class_weight": "balanced", "random_state": 42},
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune weekly classification models")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--dataset", type=str, help="Optional explicit dataset path")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--ticker", type=str, help="Restrict to single ticker dataset")
    parser.add_argument("--cv-splits", type=int, help="Override TimeSeriesSplit splits")
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
        preds, probs = model_utils.predict(estimator, X_val)
        metrics = metrics_utils.classification_metrics(y_val, preds, probs)
        fold_metrics.append(metrics)
        scores.append(metrics.get("f1_macro", 0.0))
    return float(np.mean(scores)), fold_metrics


def main() -> None:
    args = parse_args()
    config = io_utils.load_yaml(args.config)
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
            print(f"Scored model={model_name} params={params} avg_f1={avg_score:.4f}")

    if not best_spec:
        raise RuntimeError("No valid model configuration found during tuning")

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    tuning_run_dir = io_utils.path_for_run(TUNING_ROOT, f"{timestamp}_{best_spec['model']}")

    cv_df = pd.DataFrame.from_records(records)
    io_utils.save_csv(cv_df, tuning_run_dir / "cv_results.csv")
    io_utils.save_yaml(
        {
            "best_model": best_spec["model"],
            "best_params": best_spec["params"],
            "cv_metric": "f1_macro",
            "avg_score": best_score,
        },
        tuning_run_dir / "best_summary.yaml",
    )

    best_estimator = model_utils.fit_model(trainval_features, trainval_target, TASK, best_spec["model"], best_spec["params"])
    test_pred, test_prob = model_utils.predict(best_estimator, X_test)
    test_metrics = metrics_utils.classification_metrics(y_test, test_pred, test_prob)

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
        },
        final_dir / "params.yaml",
    )
    io_utils.save_json({"test": test_metrics}, final_dir / "metrics_test.json")

    notes_path = final_dir / "notes.md"
    notes_path.write_text(
        "\n".join(
            [
                f"# Weekly Classification Final Model ({timestamp})",
                f"Model: {best_spec['model']}",
                f"Parameters: {best_spec['params']}",
                f"Average tuning score (f1_macro): {best_score:.4f}",
                f"Test metrics: {json.dumps(test_metrics, indent=2)}",
                "",
                "Tuning details saved under data/results/tuning/weekly/classification/",
            ]
        ),
        encoding="utf-8",
    )

    io_utils.save_json({"test": test_metrics}, tuning_run_dir / "test_metrics.json")
    print(f"Saved weekly classification model -> {final_dir}")


if __name__ == "__main__":
    main()

