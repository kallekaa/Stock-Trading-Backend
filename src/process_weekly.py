"""Aggregate daily data into weekly features and targets."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from utils import feature_utils, io_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process weekly stock data into ML-ready datasets")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to global config file")
    parser.add_argument("--tickers", nargs="*", help="Optional subset of tickers to process")
    parser.add_argument("--format", choices=["csv", "parquet"], help="Override storage format")
    parser.add_argument("--week-ending", type=str, default="W-FRI", help="Pandas frequency alias for week aggregation")
    return parser.parse_args()


def load_raw_dataframe(raw_dir: Path, ticker: str) -> pd.DataFrame:
    parquet_path = raw_dir / f"{ticker.lower()}.parquet"
    csv_path = raw_dir / f"{ticker.lower()}.csv"
    if parquet_path.exists():
        return io_utils.load_parquet(parquet_path)
    if csv_path.exists():
        return io_utils.load_csv(csv_path)
    raise FileNotFoundError(f"Missing raw data file for {ticker} in {raw_dir}")


def resample_weekly(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    grouped = df.set_index("date").resample(freq)
    weekly = pd.DataFrame({
        "open": grouped["open"].first(),
        "high": grouped["high"].max(),
        "low": grouped["low"].min(),
        "close": grouped["close"].last(),
        "adj_close": grouped["adj_close"].last(),
        "volume": grouped["volume"].sum(),
    })
    weekly = weekly.dropna(subset=["open", "high", "low", "close"]).reset_index()
    return weekly


def process_ticker(
    df: pd.DataFrame,
    ticker: str,
    feature_cfg: dict,
    target_cfg: feature_utils.TargetConfig,
) -> pd.DataFrame:
    df = feature_utils.compose_feature_pipeline(df, feature_cfg)
    df = feature_utils.add_targets(df, target_cfg)
    df = feature_utils.finalize_features(df)
    df["ticker"] = ticker
    return df


def save_processed(df: pd.DataFrame, output_dir: Path, ticker: str, storage_format: str) -> None:
    filename = output_dir / f"{ticker.lower()}.{storage_format}"
    if storage_format == "csv":
        io_utils.save_csv(df, filename)
    else:
        io_utils.save_parquet(df, filename)


def main() -> None:
    args = parse_args()
    config = io_utils.load_yaml(args.config)
    feature_cfg = config.get("features", {}).get("weekly", {})
    target_cfg = feature_utils.TargetConfig(
        regression_horizon=config["modeling"]["horizons"]["regression"],
        classification_horizon=config["modeling"]["horizons"]["classification"],
        forecasting_horizons=config["modeling"]["horizons"]["forecasting"],
    )
    raw_dir = Path(config["data"]["raw_dir"])
    processed_dir = io_utils.ensure_directory(config["data"]["processed_weekly_dir"])
    storage_format = args.format or config["data"].get("storage_format", "parquet")

    tickers: List[str] = args.tickers or config["data"]["tickers"]
    processed_frames: list[pd.DataFrame] = []
    for ticker in tickers:
        raw_df = load_raw_dataframe(raw_dir, ticker)
        if raw_df.empty:
            print(f"Skipping {ticker}: raw dataframe empty")
            continue
        weekly = resample_weekly(raw_df, args.week_ending)
        if weekly.empty:
            print(f"Skipping {ticker}: weekly aggregation empty")
            continue
        processed_df = process_ticker(weekly, ticker, feature_cfg, target_cfg)
        if processed_df.empty:
            print(f"Skipping {ticker}: processed dataframe empty")
            continue
        save_processed(processed_df, Path(processed_dir), ticker, storage_format)
        processed_frames.append(processed_df)
        print(f"Processed weekly {ticker}: {len(processed_df)} rows")

    if processed_frames:
        combined = pd.concat(processed_frames, ignore_index=True)
        save_processed(combined, Path(processed_dir), "all", storage_format)
        print(f"Saved combined weekly dataset with {len(combined)} rows")


if __name__ == "__main__":
    main()

