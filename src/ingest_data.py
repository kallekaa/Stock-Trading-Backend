"""Download and store raw market data for configured tickers."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf

from utils import io_utils


RENAMED_COLUMNS = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest daily OHLCV data for configured tickers")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to global config file")
    parser.add_argument("--tickers", nargs="*", help="Override tickers from config")
    parser.add_argument("--start", type=str, help="Override start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="Override end date (YYYY-MM-DD)")
    parser.add_argument("--format", choices=["csv", "parquet"], help="Storage format override")
    parser.add_argument("--chunk-size", type=int, default=None, help="Optional chunk size for batched saving")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    return io_utils.load_yaml(config_path)


def normalize_dataframe(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df = df.reset_index().rename(columns=RENAMED_COLUMNS)
    df = df.rename(columns={"Date": "date"})
    df["ticker"] = ticker
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df[["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]]
    df = df.sort_values("date")
    return df


def download_data(ticker: str, start: str | None, end: str | None, tz: str | None) -> pd.DataFrame:
    data = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if data.empty:
        raise ValueError(f"No data received for ticker '{ticker}'")
    df = normalize_dataframe(data, ticker)
    if tz:
        df["date"] = df["date"].dt.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT").dt.tz_convert("UTC").dt.tz_localize(None)
    return df


def determine_output_paths(raw_dir: str, ticker: str, storage_format: str) -> Path:
    filename = f"{ticker.lower()}.{storage_format}"
    return Path(raw_dir) / filename


def save_dataframe(df: pd.DataFrame, path: Path, storage_format: str, chunk_size: int | None = None) -> None:
    if storage_format == "csv":
        io_utils.save_csv(df, path)
    else:
        if chunk_size and chunk_size < len(df):
            chunks = list(io_utils.chunk_dataframe(df, chunk_size))
            for idx, chunk in enumerate(chunks):
                chunk_path = path.with_name(f"{path.stem}_part{idx}.{path.suffix.lstrip('.')}")
                io_utils.save_parquet(chunk, chunk_path)
        else:
            io_utils.save_parquet(df, path)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    tickers = args.tickers or config["data"]["tickers"]
    start_date = args.start or config["data"].get("start_date")
    end_date = args.end or config["data"].get("end_date")
    tz = config["data"].get("timezone")
    storage_format = args.format or config["data"].get("storage_format", "parquet")
    raw_dir = io_utils.ensure_directory(config["data"]["raw_dir"])  # type: ignore[arg-type]

    for ticker in tickers:
        try:
            df = download_data(ticker, start_date, end_date, tz)
        except Exception as exc:  # pragma: no cover - network errors
            print(f"Failed to download {ticker}: {exc}")
            continue
        output_path = determine_output_paths(str(raw_dir), ticker, storage_format)
        save_dataframe(df, output_path, storage_format, args.chunk_size)
        print(f"Saved {len(df)} rows for {ticker} -> {output_path}")


if __name__ == "__main__":
    main()

