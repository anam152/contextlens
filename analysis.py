from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)


def infer_task_type(target: pd.Series) -> tuple[str, str]:
    """Infer classification or regression from target semantics."""
    clean = target.dropna()
    n = len(clean)
    unique = int(clean.nunique())

    if n == 0:
        return "classification", "the target contains no observed values"

    if (
        is_object_dtype(clean)
        or is_string_dtype(clean)
        or isinstance(clean.dtype, pd.CategoricalDtype)
        or is_bool_dtype(clean)
    ):
        return "classification", "the target is categorical or textual"

    classification_threshold = max(10, min(30, int(np.sqrt(n)) + 1))
    if is_integer_dtype(clean) and unique <= classification_threshold:
        return (
            "classification",
            f"the integer target has only {unique} distinct values",
        )

    unique_ratio = unique / n
    if unique <= 20 and unique_ratio <= 0.10:
        return (
            "classification",
            f"the target has {unique} repeated discrete values",
        )

    if is_numeric_dtype(clean):
        return "regression", "the target is numeric with many distinct values"

    return "classification", "the target is non-continuous"


def _is_datetime_like(series: pd.Series, column_name: str = "") -> bool:
    if is_datetime64_any_dtype(series):
        return True
    if not (is_object_dtype(series) or is_string_dtype(series)):
        return False

    sample = series.dropna().astype(str).head(100)
    if sample.empty:
        return False

    name = column_name.lower()
    name_signal = any(
        token in name
        for token in ("date", "time", "timestamp", "created", "updated")
    )
    content_signal = sample.str.contains(
        r"(?:\d{4}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
        regex=True,
    ).mean() >= 0.50
    if not (name_signal or content_signal):
        return False

    parsed = pd.to_datetime(sample, errors="coerce")
    return bool(parsed.notna().mean() >= 0.80)


def _likely_identifier(series: pd.Series, column_name: str) -> bool:
    clean = series.dropna()
    if clean.empty:
        return False

    name = column_name.lower().strip()
    name_signal = (
        name == "id"
        or name.endswith("_id")
        or name.startswith("id_")
        or any(token in name for token in ("uuid", "identifier", "record_number"))
    )
    uniqueness = clean.nunique() / len(clean)
    textual_high_uniqueness = (
        (is_object_dtype(clean) or is_string_dtype(clean))
        and uniqueness >= 0.98
    )
    return bool(name_signal or textual_high_uniqueness)


def analyse_dataset(
    df: pd.DataFrame,
    target: str,
    task: str,
) -> dict[str, Any]:
    """Profile a tabular dataset and create transparent context flags."""
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' does not exist.")
    if task not in {"classification", "regression"}:
        raise ValueError("task must be 'classification' or 'regression'.")

    feature_df = df.drop(columns=[target])

    datetime_like = [
        column for column in feature_df.columns if _is_datetime_like(feature_df[column], str(column))
    ]
    numeric = [
        column
        for column in feature_df.select_dtypes(include=np.number).columns
        if column not in datetime_like
    ]
    categorical = [
        column
        for column in feature_df.columns
        if column not in numeric and column not in datetime_like
    ]

    missing_by_column = df.isna().sum().astype(int)
    missing_by_column = {
        str(column): int(count)
        for column, count in missing_by_column.items()
        if count > 0
    }

    constant_columns = [
        column for column in feature_df.columns if feature_df[column].nunique(dropna=False) <= 1
    ]
    id_columns = [
        column
        for column in feature_df.columns
        if column not in datetime_like
        and _likely_identifier(feature_df[column], str(column))
    ]
    high_cardinality = [
        column
        for column in categorical
        if feature_df[column].nunique(dropna=True) > 50
        and feature_df[column].nunique(dropna=True) / max(len(feature_df), 1) > 0.20
    ]

    issues: list[dict[str, str]] = []

    target_missing = int(df[target].isna().sum())
    if target_missing:
        issues.append(
            {
                "severity": "high",
                "title": "Missing target values",
                "detail": f"{target_missing:,} rows have no observed target.",
                "recommendation": "Exclude these rows from supervised training or recover the labels.",
            }
        )

    total_missing = int(df.isna().sum().sum())
    if total_missing:
        missing_rate = total_missing / max(df.size, 1)
        issues.append(
            {
                "severity": "medium" if missing_rate < 0.20 else "high",
                "title": "Missing feature values",
                "detail": f"{total_missing:,} cells are missing ({missing_rate:.1%} of the table).",
                "recommendation": "Use fold-safe imputation and inspect whether missingness is informative.",
            }
        )

    duplicates = int(df.duplicated().sum())
    if duplicates:
        issues.append(
            {
                "severity": "medium",
                "title": "Duplicate rows",
                "detail": f"{duplicates:,} exact duplicate rows were detected.",
                "recommendation": "Confirm whether duplicates are repeated observations or data-entry artefacts.",
            }
        )

    if constant_columns:
        issues.append(
            {
                "severity": "low",
                "title": "Constant features",
                "detail": f"{len(constant_columns)} columns contain no predictive variation: {', '.join(map(str, constant_columns[:8]))}.",
                "recommendation": "Remove constant features before final modelling.",
            }
        )

    if id_columns:
        issues.append(
            {
                "severity": "medium",
                "title": "Possible identifier leakage",
                "detail": f"Potential identifiers: {', '.join(map(str, id_columns[:8]))}.",
                "recommendation": "Exclude row identifiers unless they encode legitimate domain information.",
            }
        )

    if high_cardinality:
        issues.append(
            {
                "severity": "medium",
                "title": "High-cardinality categories",
                "detail": f"Potentially expensive categorical columns: {', '.join(map(str, high_cardinality[:8]))}.",
                "recommendation": "Consider grouping rare categories, hashing, or domain-specific encodings.",
            }
        )

    class_distribution = None
    imbalance_ratio = None
    if task == "classification":
        counts = df[target].dropna().value_counts()
        class_distribution = {
            str(label): int(count) for label, count in counts.items()
        }
        if len(counts) < 2:
            issues.append(
                {
                    "severity": "high",
                    "title": "Single-class target",
                    "detail": "Only one observed target class is available.",
                    "recommendation": "Collect or restore examples from at least one additional class.",
                }
            )
        elif counts.min() > 0:
            imbalance_ratio = float(counts.max() / counts.min())
            if imbalance_ratio >= 3:
                issues.append(
                    {
                        "severity": "high" if imbalance_ratio >= 10 else "medium",
                        "title": "Class imbalance",
                        "detail": f"The largest class is {imbalance_ratio:.1f}× the size of the smallest class.",
                        "recommendation": "Prioritise macro-F1 and per-class recall; use stratified validation.",
                    }
                )

    if len(df) < 200:
        issues.append(
            {
                "severity": "medium",
                "title": "Small sample",
                "detail": f"The dataset contains only {len(df):,} rows.",
                "recommendation": "Treat a single hold-out score as uncertain; use repeated or nested cross-validation in formal work.",
            }
        )

    if len(feature_df.columns) >= max(50, len(df) // 2):
        issues.append(
            {
                "severity": "medium",
                "title": "High-dimensional setting",
                "detail": f"There are {len(feature_df.columns):,} features for {len(df):,} observations.",
                "recommendation": "Use regularisation, feature selection, and leakage-safe validation.",
            }
        )

    return {
        "shape": {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "features": int(len(feature_df.columns)),
        },
        "target": target,
        "task": task,
        "columns": {
            "numeric": list(map(str, numeric)),
            "categorical": list(map(str, categorical)),
            "datetime_like": list(map(str, datetime_like)),
            "constant": list(map(str, constant_columns)),
            "likely_identifiers": list(map(str, id_columns)),
            "high_cardinality": list(map(str, high_cardinality)),
        },
        "missing": {
            "total_cells": total_missing,
            "by_column": missing_by_column,
            "target_missing": target_missing,
        },
        "duplicates": duplicates,
        "class_distribution": class_distribution,
        "imbalance_ratio": imbalance_ratio,
        "issues": issues,
    }


def build_context_narrative(analysis: dict[str, Any]) -> dict[str, Any]:
    """Turn the structured profile into concise, auditable language."""
    rows = analysis["shape"]["rows"]
    features = analysis["shape"]["features"]
    task = analysis["task"]
    columns = analysis["columns"]

    observations = [
        f"The selected target defines a {task} problem.",
        f"The modelling table contains {rows:,} observations and {features:,} candidate features.",
        f"{len(columns['numeric'])} features are numeric and {len(columns['categorical'])} are categorical.",
    ]

    if columns["datetime_like"]:
        observations.append(
            f"{len(columns['datetime_like'])} columns look date- or time-like and require temporal interpretation."
        )
    if columns["likely_identifiers"]:
        observations.append(
            "One or more highly unique columns may identify records rather than generalisable patterns."
        )
    if analysis["missing"]["total_cells"]:
        observations.append(
            "Missingness is part of the dataset context and is handled inside each training pipeline."
        )

    if task == "classification":
        distribution = analysis["class_distribution"] or {}
        observations.append(f"The target contains {len(distribution)} observed classes.")
        evaluation_guidance = [
            "Use stratified splitting whenever class counts permit it.",
            "Use macro-F1 as the primary comparison metric so minority classes matter.",
            "Inspect per-class precision and recall rather than relying on accuracy alone.",
        ]
        if analysis["imbalance_ratio"] and analysis["imbalance_ratio"] >= 3:
            evaluation_guidance.insert(
                0,
                "The class distribution is imbalanced; accuracy can conceal poor minority-class performance.",
            )
    else:
        evaluation_guidance = [
            "Use RMSE as the primary comparison metric because large errors receive more weight.",
            "Inspect MAE alongside RMSE to understand typical absolute error.",
            "Interpret R² relative to a simple baseline and the target's natural variability.",
        ]

    if rows < 200:
        evaluation_guidance.append(
            "Because the sample is small, use cross-validation before making publication-level claims."
        )

    summary = (
        f"ContextLens interprets this as a **{task}** task with "
        f"**{rows:,} rows** and **{features:,} candidate features**. "
        f"It detected **{len(analysis['issues'])} contextual issue(s)** requiring review."
    )

    return {
        "summary": summary,
        "observations": observations,
        "evaluation_guidance": evaluation_guidance,
    }
