"""Generate classification predictions using the finalized daily model."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from utils import io_utils

FREQUENCY = "daily"
TASK = "classification"
TARGET_COLUMN = "target_classification_direction"
FINAL_MODEL_DIR = Path("data/results/final_models/daily/classification")
PREDICTION_DIR = Path("data/predictions/daily/classification")
DROP_COLUMNS = ["date", "ticker"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run daily classification inference")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--dataset", type=str, help="Optional explicit dataset path")
    parser.add_argument("--recent", type=int, help="Limit to most recent N rows")
    parser.add_argument("--ticker", type=str, help="Restrict to single ticker dataset")
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


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [col for col in DROP_COLUMNS if col in df.columns]
    return df.drop(columns=cols_to_drop)


def main() -> None:
    args = parse_args()
    config = io_utils.load_yaml(args.config)
    params = io_utils.load_yaml(FINAL_MODEL_DIR / "params.yaml")
    model = io_utils.load_model(FINAL_MODEL_DIR / "model.pkl")

    df = load_dataset(args, config)
    df = df.sort_values("date").reset_index(drop=True)
    if args.recent:
        df = df.tail(args.recent)
    else:
        recent = config.get("predict", {}).get("recent_periods")
        if recent:
            df = df.tail(int(recent))

    features = prepare_features(df)
    predictions = model.predict(features)
    probs = model.predict_proba(features) if hasattr(model, "predict_proba") else None

    output_df = pd.DataFrame(
        {
            "date": df["date"],
            "ticker": df.get("ticker", pd.Series([""] * len(df))),
            "prediction": predictions.astype(int),
        }
    )
    if TARGET_COLUMN in df.columns:
        output_df["actual_label"] = df[TARGET_COLUMN].astype(int)

    if probs is not None:
        prob_df = pd.DataFrame(probs, columns=[f"prob_class_{i}" for i in range(probs.shape[1])])
        output_df = pd.concat([output_df.reset_index(drop=True), prob_df], axis=1)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = io_utils.ensure_directory(PREDICTION_DIR)
    output_path = out_dir / f"predictions_{timestamp}.csv"
    io_utils.save_csv(output_df, output_path)

    summary = {
        "task": TASK,
        "frequency": FREQUENCY,
        "model": params.get("model"),
        "params": params.get("params"),
        "rows": len(output_df),
        "output_path": str(output_path),
    }
    io_utils.save_json(summary, out_dir / f"metadata_{timestamp}.json")
    print(f"Saved classification predictions -> {output_path}")


if __name__ == "__main__":
    main()

