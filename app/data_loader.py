from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd

from app.config import (
    DATASET_CACHE_FILE,
    DATASET_NAME,
    REQUIRED_ANALYSIS_COLUMNS,
    ensure_runtime_dirs,
)


_COLUMN_ALIASES: dict[str, str] = {
    "instruction": "instruction",
    "instructions": "instruction",
    "query": "instruction",
    "question": "instruction",
    "text": "instruction",
    "customer_request": "instruction",
    "response": "response",
    "answer": "response",
    "completion": "response",
    "category": "category",
    "label": "category",
    "intent": "intent",
}


def normalize_column_name(column_name: str) -> str:
    """Normalize a raw dataset column name into a stable snake-case name."""
    normalized = (
        str(column_name)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("/", "_")
    )

    while "__" in normalized:
        normalized = normalized.replace("__", "_")

    return _COLUMN_ALIASES.get(normalized, normalized)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the DataFrame with normalized column names."""
    normalized = df.copy()
    normalized.columns = [normalize_column_name(column) for column in normalized.columns]
    return normalized


def add_stable_row_id(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the DataFrame has a stable integer row_id column."""
    normalized = df.copy()

    if "row_id" in normalized.columns:
        normalized["row_id"] = normalized["row_id"].astype(int)
        return normalized

    normalized.insert(0, "row_id", range(len(normalized)))
    return normalized


def _load_from_huggingface() -> pd.DataFrame:
    """Load the Bitext dataset from Hugging Face datasets."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required to download the Bitext dataset. "
            "Install project dependencies first."
        ) from exc

    dataset = load_dataset(DATASET_NAME)

    if "train" in dataset:
        split = dataset["train"]
    else:
        first_split_name = next(iter(dataset.keys()))
        split = dataset[first_split_name]

    return split.to_pandas()


def _load_from_cache() -> pd.DataFrame | None:
    """Load the local cached CSV if it exists."""
    if not DATASET_CACHE_FILE.exists():
        return None

    return pd.read_csv(DATASET_CACHE_FILE)


def _save_cache(df: pd.DataFrame) -> None:
    """Persist the normalized dataset locally for repeatable offline runs."""
    ensure_runtime_dirs()
    df.to_csv(DATASET_CACHE_FILE, index=False)


def _ensure_analysis_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure expected analysis columns exist, filling unavailable ones with None."""
    normalized = df.copy()

    for column in REQUIRED_ANALYSIS_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = None

    return normalized


def load_dataset_df(force_reload: bool = False) -> pd.DataFrame:
    """Load, normalize, and cache the Bitext Customer Service dataset."""
    ensure_runtime_dirs()

    cached_df = None if force_reload else _load_from_cache()

    if cached_df is not None:
        raw_df = cached_df
    else:
        raw_df = _load_from_huggingface()

    normalized_df = normalize_columns(raw_df)
    normalized_df = add_stable_row_id(normalized_df)
    normalized_df = _ensure_analysis_columns(normalized_df)

    if cached_df is None or force_reload:
        _save_cache(normalized_df)

    return normalized_df


@lru_cache(maxsize=1)
def get_dataset_df() -> pd.DataFrame:
    """Return a cached normalized dataset DataFrame."""
    return load_dataset_df(force_reload=False)


def get_dataset_metadata(include_sample_values: bool = True) -> dict[str, Any]:
    """Return dataset metadata with available columns and optional sample values."""
    df = get_dataset_df()

    sample_values: dict[str, list[str]] | None = None
    if include_sample_values:
        sample_values = {}
        for column in df.columns:
            values = (
                df[column]
                .dropna()
                .astype(str)
                .drop_duplicates()
                .head(5)
                .tolist()
            )
            sample_values[column] = values

    return {
        "columns": list(df.columns),
        "row_count": int(len(df)),
        "sample_values": sample_values,
    }