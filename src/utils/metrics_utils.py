"""Metrics helper functions for experiments and evaluation."""
from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd
from sklearn import metrics


def regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> Dict[str, float]:
    y_true = np.asarray(list(y_true))
    y_pred = np.asarray(list(y_pred))
    return {
        "mse": metrics.mean_squared_error(y_true, y_pred),
        "rmse": metrics.mean_squared_error(y_true, y_pred, squared=False),
        "mae": metrics.mean_absolute_error(y_true, y_pred),
        "mape": metrics.mean_absolute_percentage_error(y_true, y_pred),
        "r2": metrics.r2_score(y_true, y_pred),
    }


def classification_metrics(y_true: Iterable[int], y_pred: Iterable[int], y_prob: Iterable[float] | None = None) -> Dict[str, float]:
    y_true = np.asarray(list(y_true))
    y_pred = np.asarray(list(y_pred))
    results = {
        "accuracy": metrics.accuracy_score(y_true, y_pred),
        "precision_macro": metrics.precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": metrics.recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": metrics.f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    if y_prob is not None:
        try:
            results["roc_auc_ovr"] = metrics.roc_auc_score(y_true, y_prob, multi_class="ovr")
        except ValueError:
            pass
    return results


def forecasting_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> Dict[str, float]:
    return regression_metrics(y_true, y_pred)


def summarize_predictions(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    residuals = y_true - y_pred
    return {
        "residual_mean": float(residuals.mean()),
        "residual_std": float(residuals.std()),
        "residual_min": float(residuals.min()),
        "residual_max": float(residuals.max()),
    }

