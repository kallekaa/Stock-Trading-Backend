"""Utilities for chronological dataset splitting."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Iterable, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


DATE_COL = "date"


@dataclass
class SplitResult:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.DataFrame | pd.Series
    y_val: pd.DataFrame | pd.Series
    y_test: pd.DataFrame | pd.Series


def _extract_target(df: pd.DataFrame, target_cols: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    y = df[list(target_cols)] if target_cols else pd.DataFrame()
    X = df.drop(columns=list(target_cols)) if target_cols else df.copy()
    return X, y


def temporal_train_val_test_split(
    df: pd.DataFrame,
    target_cols: Sequence[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> SplitResult:
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]
    X_train, y_train = _extract_target(train, target_cols)
    X_val, y_val = _extract_target(val, target_cols)
    X_test, y_test = _extract_target(test, target_cols)
    return SplitResult(X_train, X_val, X_test, y_train, y_val, y_test)


def walk_forward_splits(
    df: pd.DataFrame,
    n_splits: int,
    window_size: int | None = None,
    test_size: int | None = None,
) -> Generator[tuple[pd.DataFrame, pd.DataFrame], None, None]:
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    if window_size is None or test_size is None:
        splitter = TimeSeriesSplit(n_splits=n_splits)
        for train_idx, test_idx in splitter.split(df):
            yield df.iloc[train_idx], df.iloc[test_idx]
    else:
        max_start = len(df) - (window_size + test_size)
        if max_start <= 0:
            raise ValueError("Not enough observations for requested walk-forward setup")
        step = max(1, (max_start // n_splits) or 1)
        for start in range(0, max_start + 1, step):
            train_slice = df.iloc[start : start + window_size]
            test_slice = df.iloc[start + window_size : start + window_size + test_size]
            if len(test_slice) < test_size:
                break
            yield train_slice, test_slice


def rolling_windows(
    df: pd.DataFrame,
    window_size: int,
    step: int,
) -> Iterable[pd.DataFrame]:
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    for start in range(0, len(df) - window_size + 1, step):
        yield df.iloc[start : start + window_size]


def align_targets(
    X: pd.DataFrame,
    y: pd.DataFrame | pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame | pd.Series]:
    aligned = X.join(y, how="inner") if isinstance(y, pd.DataFrame) else X.join(y, how="inner")
    target_cols = list(y.columns) if isinstance(y, pd.DataFrame) else [y.name]
    X_aligned = aligned.drop(columns=target_cols, errors="ignore")
    y_aligned = aligned[target_cols] if target_cols else pd.DataFrame()
    if isinstance(y, pd.Series):
        y_aligned = y_aligned[target_cols[0]]
    return X_aligned, y_aligned

