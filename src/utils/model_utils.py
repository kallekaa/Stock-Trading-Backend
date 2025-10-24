"""Common model helpers for experiments and tuning."""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MODEL_REGISTRY = {
    "regression": {
        "LinearRegression": LinearRegression,
        "RandomForestRegressor": RandomForestRegressor,
        "GradientBoostingRegressor": GradientBoostingRegressor,
    },
    "classification": {
        "LogisticRegression": LogisticRegression,
        "RandomForestClassifier": RandomForestClassifier,
    },
    "time_series": {
        "RandomForestRegressor": RandomForestRegressor,
        "GradientBoostingRegressor": GradientBoostingRegressor,
        "LinearRegression": LinearRegression,
    },
}


def create_model(task: str, model_name: str, params: Dict[str, Any] | None = None) -> BaseEstimator:
    params = params or {}
    if task not in MODEL_REGISTRY:
        raise KeyError(f"Unsupported task: {task}")
    if model_name not in MODEL_REGISTRY[task]:
        raise KeyError(f"Unsupported model '{model_name}' for task '{task}'")
    model_cls = MODEL_REGISTRY[task][model_name]
    estimator: BaseEstimator
    if task in {"regression", "time_series"} and model_name == "LinearRegression":
        estimator = Pipeline([("scaler", StandardScaler()), ("model", model_cls(**params))])
    elif task == "classification" and model_name == "LogisticRegression":
        estimator = Pipeline([("scaler", StandardScaler()), ("model", model_cls(**params))])
    else:
        estimator = model_cls(**params)
    return estimator


def fit_model(
    X_train: pd.DataFrame,
    y_train: pd.Series | pd.DataFrame,
    task: str,
    model_name: str,
    params: Dict[str, Any] | None = None,
) -> BaseEstimator:
    estimator = create_model(task, model_name, params)
    if isinstance(y_train, pd.DataFrame) and y_train.shape[1] == 1:
        y_train = y_train.iloc[:, 0]
    estimator.fit(X_train, y_train)
    return estimator


def predict(estimator: BaseEstimator, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray | None]:
    if hasattr(estimator, "predict_proba"):
        probs = estimator.predict_proba(X)
    else:
        probs = None
    preds = estimator.predict(X)
    return preds, probs

