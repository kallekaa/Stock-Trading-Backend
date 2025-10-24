"""Feature engineering helpers for daily and weekly datasets."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


PRICE_COL = "close"
DATE_COL = "date"
TICKER_COL = "ticker"


@dataclass
class TargetConfig:
    regression_horizon: int
    classification_horizon: int
    forecasting_horizons: Sequence[int]
    neutral_return_threshold: float = 0.001


def compute_log_return(series: pd.Series, periods: int = 1) -> pd.Series:
    return np.log(series / series.shift(periods))


def rolling_mean(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def rolling_std(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).std()


def rolling_min(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).min()


def rolling_max(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).max()


def exponential_moving_average(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def relative_strength_index(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(gains, index=series.index).rolling(window=window, min_periods=window).mean()
    roll_down = pd.Series(losses, index=series.index).rolling(window=window, min_periods=window).mean()
    rs = roll_up / (roll_down + 1e-9)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = exponential_moving_average(series, span=fast)
    ema_slow = exponential_moving_average(series, span=slow)
    macd_line = ema_fast - ema_slow
    signal_line = exponential_moving_average(macd_line, span=signal)
    histogram = macd_line - signal_line
    return pd.DataFrame({
        "macd": macd_line,
        "macd_signal": signal_line,
        "macd_hist": histogram,
    })


def add_price_features(df: pd.DataFrame, windows: Sequence[int]) -> pd.DataFrame:
    df = df.copy()
    for window in windows:
        df[f"sma_{window}"] = rolling_mean(df[PRICE_COL], window)
        df[f"ema_{window}"] = exponential_moving_average(df[PRICE_COL], span=window)
        df[f"return_{window}"] = df[PRICE_COL].pct_change(periods=window)
        df[f"log_return_{window}"] = compute_log_return(df[PRICE_COL], periods=window)
    return df


def add_volatility_features(df: pd.DataFrame, windows: Sequence[int]) -> pd.DataFrame:
    df = df.copy()
    for window in windows:
        df[f"volatility_{window}"] = rolling_std(df[PRICE_COL], window)
        df[f"range_{window}"] = rolling_max(df["high"], window) - rolling_min(df["low"], window)
        df[f"rolling_volume_{window}"] = rolling_mean(df["volume"], window)
    return df


def add_momentum_features(df: pd.DataFrame, windows: Sequence[int]) -> pd.DataFrame:
    df = df.copy()
    for window in windows:
        df[f"momentum_{window}"] = df[PRICE_COL] / df[PRICE_COL].shift(window) - 1
    return df


def add_rsi_features(df: pd.DataFrame, windows: Sequence[int]) -> pd.DataFrame:
    df = df.copy()
    for window in windows:
        df[f"rsi_{window}"] = relative_strength_index(df[PRICE_COL], window)
    return df


def add_macd_features(df: pd.DataFrame, macd_config: dict[str, int]) -> pd.DataFrame:
    df = df.copy()
    macd_frame = macd(df[PRICE_COL], **macd_config)
    return pd.concat([df, macd_frame], axis=1)


def compose_feature_pipeline(df: pd.DataFrame, feature_cfg: dict[str, any]) -> pd.DataFrame:
    df = df.sort_values(DATE_COL).copy()
    df = add_price_features(df, feature_cfg.get("price_windows", []))
    df = add_volatility_features(df, feature_cfg.get("volatility_windows", []))
    df = add_momentum_features(df, feature_cfg.get("momentum_windows", []))
    df = add_rsi_features(df, feature_cfg.get("rsi_windows", []))
    macd_config = feature_cfg.get("macd")
    if macd_config:
        df = add_macd_features(df, macd_config)
    df["log_volume"] = np.log1p(df["volume"].replace(0, np.nan)).fillna(0.0)
    return df


def add_targets(df: pd.DataFrame, target_cfg: TargetConfig) -> pd.DataFrame:
    df = df.copy()
    df["target_regression_return"] = (
        df[PRICE_COL]
        .pct_change(periods=target_cfg.regression_horizon)
        .shift(-target_cfg.regression_horizon)
    )
    future_return = (
        df[PRICE_COL]
        .pct_change(periods=target_cfg.classification_horizon)
        .shift(-target_cfg.classification_horizon)
    )
    neutral = target_cfg.neutral_return_threshold
    df["target_classification_direction"] = np.select(
        [future_return > neutral, future_return < -neutral],
        [2, 0],
        default=1,
    )
    for horizon in target_cfg.forecasting_horizons:
        df[f"target_forecast_close_t+{horizon}"] = df[PRICE_COL].shift(-horizon)
    return df


def finalize_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(DATE_COL).dropna().reset_index(drop=True)
    return df

