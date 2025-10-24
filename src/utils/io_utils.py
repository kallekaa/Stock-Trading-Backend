"""Utility helpers for file IO and serialization."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import joblib
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENCODING = "utf-8"


def ensure_directory(path: Path | str) -> Path:
    """Create the directory (and parents) if it does not already exist."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def resolve_path(path: Path | str) -> Path:
    """Return an absolute path relative to the project root when needed."""
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def load_csv(path: Path | str, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(resolve_path(path), **kwargs)


def save_csv(df: pd.DataFrame, path: Path | str, **kwargs: Any) -> Path:
    target = resolve_path(path)
    ensure_directory(target.parent)
    df.to_csv(target, index=False, **kwargs)
    return target


def load_parquet(path: Path | str, **kwargs: Any) -> pd.DataFrame:
    return pd.read_parquet(resolve_path(path), **kwargs)


def save_parquet(df: pd.DataFrame, path: Path | str, **kwargs: Any) -> Path:
    target = resolve_path(path)
    ensure_directory(target.parent)
    df.to_parquet(target, index=False, **kwargs)
    return target


def save_json(data: Dict[str, Any], path: Path | str, **kwargs: Any) -> Path:
    target = resolve_path(path)
    ensure_directory(target.parent)
    with target.open("w", encoding=DEFAULT_ENCODING) as fh:
        json.dump(data, fh, **({"indent": 2} | kwargs))
    return target


def load_json(path: Path | str) -> Dict[str, Any]:
    with resolve_path(path).open("r", encoding=DEFAULT_ENCODING) as fh:
        return json.load(fh)


def save_yaml(data: Dict[str, Any], path: Path | str) -> Path:
    target = resolve_path(path)
    ensure_directory(target.parent)
    with target.open("w", encoding=DEFAULT_ENCODING) as fh:
        yaml.safe_dump(data, fh, sort_keys=False)
    return target


def load_yaml(path: Path | str) -> Dict[str, Any]:
    with resolve_path(path).open("r", encoding=DEFAULT_ENCODING) as fh:
        return yaml.safe_load(fh)


def save_model(model: Any, path: Path | str) -> Path:
    target = resolve_path(path)
    ensure_directory(target.parent)
    joblib.dump(model, target)
    return target


def load_model(path: Path | str) -> Any:
    return joblib.load(resolve_path(path))


def list_files(directory: Path | str, patterns: Optional[Iterable[str]] = None) -> Iterable[Path]:
    directory = resolve_path(directory)
    if patterns is None:
        yield from directory.glob("*")
    else:
        for pattern in patterns:
            yield from directory.glob(pattern)


def path_for_run(root: Path | str, *components: str) -> Path:
    """Build a directory path for experimental runs and ensure it exists."""
    path = resolve_path(root)
    for component in components:
        path = path / component
    return ensure_directory(path)


def chunk_dataframe(df: pd.DataFrame, chunk_size: int) -> Iterable[pd.DataFrame]:
    for start in range(0, len(df), chunk_size):
        yield df.iloc[start : start + chunk_size]

