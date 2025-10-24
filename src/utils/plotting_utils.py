"""Plotting helpers to visualize model diagnostics."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics

from .io_utils import ensure_directory


sns.set_style("whitegrid")


def _prep_path(path: Path | str) -> Path:
    path = Path(path)
    ensure_directory(path.parent)
    return path


def plot_actual_vs_predicted(y_true: Iterable[float], y_pred: Iterable[float], path: Path | str) -> Path:
    path = _prep_path(path)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color="red", linestyle="--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def plot_residuals(y_true: Iterable[float], y_pred: Iterable[float], path: Path | str) -> Path:
    path = _prep_path(path)
    residuals = np.asarray(list(y_true)) - np.asarray(list(y_pred))
    plt.figure(figsize=(6, 4))
    sns.histplot(residuals, kde=True)
    plt.title("Residual Distribution")
    plt.xlabel("Residual")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def plot_feature_importance(importances: Iterable[float], feature_names: Iterable[str], path: Path | str) -> Path:
    path = _prep_path(path)
    values = np.asarray(list(importances))
    names = list(feature_names)
    order = np.argsort(values)[::-1]
    plt.figure(figsize=(8, 6))
    sns.barplot(x=values[order], y=np.array(names)[order])
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def plot_confusion_matrix(y_true: Iterable[int], y_pred: Iterable[int], path: Path | str) -> Path:
    path = _prep_path(path)
    cm = metrics.confusion_matrix(list(y_true), list(y_pred))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def plot_roc_curve(y_true: Iterable[int], y_prob: np.ndarray, path: Path | str) -> Path:
    path = _prep_path(path)
    if y_prob.ndim == 1:
        fpr, tpr, _ = metrics.roc_curve(y_true, y_prob)
        auc = metrics.auc(fpr, tpr)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"ROC curve (AUC={auc:.3f})")
    else:
        plt.figure(figsize=(6, 5))
        for cls in range(y_prob.shape[1]):
            fpr, tpr, _ = metrics.roc_curve((np.array(list(y_true)) == cls).astype(int), y_prob[:, cls])
            auc = metrics.auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"Class {cls} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path

