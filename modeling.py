from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


@dataclass(frozen=True)
class ModelSpec:
    factory: Callable[[int], BaseEstimator]
    summary: str


CLASSIFICATION_MODELS: dict[str, ModelSpec] = {
    "Logistic Regression": ModelSpec(
        lambda seed: LogisticRegression(
            max_iter=2_000,
            class_weight="balanced",
            random_state=seed,
        ),
        "A regularised linear baseline with balanced class weighting.",
    ),
    "Decision Tree": ModelSpec(
        lambda seed: DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=seed,
        ),
        "A non-linear tree baseline with constrained depth for interpretability.",
    ),
    "Random Forest": ModelSpec(
        lambda seed: RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        ),
        "An ensemble baseline that captures non-linear interactions.",
    ),
}

REGRESSION_MODELS: dict[str, ModelSpec] = {
    "Ridge Regression": ModelSpec(
        lambda seed: Ridge(alpha=1.0),
        "A regularised linear regression baseline.",
    ),
    "Decision Tree": ModelSpec(
        lambda seed: DecisionTreeRegressor(
            max_depth=8,
            min_samples_leaf=3,
            random_state=seed,
        ),
        "A constrained non-linear regression tree.",
    ),
    "Random Forest": ModelSpec(
        lambda seed: RandomForestRegressor(
            n_estimators=250,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        ),
        "An ensemble regression baseline for non-linear relationships.",
    ),
}


def available_models(task: str) -> dict[str, str]:
    registry = _registry(task)
    return {name: spec.summary for name, spec in registry.items()}


def _registry(task: str) -> dict[str, ModelSpec]:
    if task == "classification":
        return CLASSIFICATION_MODELS
    if task == "regression":
        return REGRESSION_MODELS
    raise ValueError("task must be 'classification' or 'regression'.")


def _drop_unusable_columns(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    dropped: list[str] = []
    kept = []

    for column in X.columns:
        series = X[column]
        if series.nunique(dropna=False) <= 1:
            dropped.append(str(column))
            continue

        clean = series.dropna()
        if not clean.empty:
            uniqueness = clean.nunique() / len(clean)
            name = str(column).lower()
            textual_high_uniqueness = (
                (
                    pd.api.types.is_object_dtype(clean)
                    or pd.api.types.is_string_dtype(clean)
                )
                and uniqueness >= 0.995
            )
            likely_id = (
                name == "id"
                or name.endswith("_id")
                or name.startswith("id_")
                or "uuid" in name
                or "identifier" in name
                or textual_high_uniqueness
            )
            if likely_id:
                dropped.append(str(column))
                continue
        kept.append(column)

    if not kept:
        raise ValueError("No usable features remain after removing constants and likely identifiers.")

    return X[kept].copy(), dropped


def _prepare_feature_types(
    X: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    X = X.copy()
    converted_datetime: list[str] = []

    for column in X.columns:
        if pd.api.types.is_datetime64_any_dtype(X[column]):
            converted_datetime.append(str(column))
        elif pd.api.types.is_object_dtype(X[column]) or pd.api.types.is_string_dtype(X[column]):
            sample = X[column].dropna().astype(str).head(100)
            if not sample.empty:
                name = str(column).lower()
                name_signal = any(
                    token in name
                    for token in ("date", "time", "timestamp", "created", "updated")
                )
                content_signal = sample.str.contains(
                    r"(?:\d{4}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
                    regex=True,
                ).mean() >= 0.50
                if name_signal or content_signal:
                    parsed = pd.to_datetime(sample, errors="coerce")
                    if parsed.notna().mean() >= 0.80:
                        converted_datetime.append(str(column))

    for column in converted_datetime:
        parsed = pd.to_datetime(X[column], errors="coerce")
        X[f"{column}__year"] = parsed.dt.year
        X[f"{column}__month"] = parsed.dt.month
        X[f"{column}__day"] = parsed.dt.day
        X[f"{column}__dayofweek"] = parsed.dt.dayofweek
        X = X.drop(columns=[column])

    numeric = list(X.select_dtypes(include=np.number).columns)
    categorical = [column for column in X.columns if column not in numeric]
    return X, numeric, categorical, converted_datetime


def _build_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> ColumnTransformer:
    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_columns))

    if categorical_columns:
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent", add_indicator=True)),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        min_frequency=2,
                        sparse_output=True,
                    ),
                ),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    if not transformers:
        raise ValueError("No numeric or categorical features are available.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def _sample_rows(
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    row_limit: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series, str | None]:
    if len(X) <= row_limit:
        return X, y, None

    if task == "classification":
        joined = X.copy()
        joined["__target__"] = y.values
        sampled_parts = []
        proportions = joined["__target__"].value_counts(normalize=True, dropna=False)
        for label, proportion in proportions.items():
            group = joined[joined["__target__"] == label]
            n = max(1, min(len(group), round(row_limit * proportion)))
            sampled_parts.append(group.sample(n=n, random_state=random_state))
        sampled = (
            pd.concat(sampled_parts)
            .sample(frac=1, random_state=random_state)
            .head(row_limit)
        )
        return (
            sampled.drop(columns=["__target__"]),
            sampled["__target__"],
            f"Training was limited to a stratified sample of {row_limit:,} rows.",
        )

    indices = X.sample(n=row_limit, random_state=random_state).index
    return (
        X.loc[indices],
        y.loc[indices],
        f"Training was limited to a random sample of {row_limit:,} rows.",
    )


def _feature_importance(
    pipeline: Pipeline,
    top_n: int = 20,
) -> list[dict[str, Any]] | None:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        return None

    raw_importance = None
    if hasattr(model, "feature_importances_"):
        raw_importance = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        coefficients = np.asarray(model.coef_)
        if coefficients.ndim == 1:
            raw_importance = np.abs(coefficients)
        else:
            raw_importance = np.mean(np.abs(coefficients), axis=0)

    if raw_importance is None or len(raw_importance) != len(feature_names):
        return None

    clean_names = [
        str(name).replace("numeric__", "").replace("categorical__", "")
        for name in feature_names
    ]
    frame = pd.DataFrame(
        {"feature": clean_names, "importance": raw_importance.astype(float)}
    )
    frame = frame.sort_values("importance", ascending=False).head(top_n)
    return frame.to_dict(orient="records")


def train_and_compare_models(
    df: pd.DataFrame,
    target: str,
    task: str,
    model_names: list[str],
    test_size: float = 0.25,
    random_state: int = 42,
    row_limit: int = 20_000,
) -> dict[str, Any]:
    """Train leakage-safe baseline pipelines and return serialisable diagnostics."""
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' does not exist.")
    if not model_names:
        raise ValueError("Select at least one model.")

    working = df.dropna(subset=[target]).copy()
    if len(working) < 20:
        raise ValueError("At least 20 rows with observed target values are required.")

    X = working.drop(columns=[target])
    y = working[target]

    if task == "classification" and y.nunique() < 2:
        raise ValueError("Classification requires at least two target classes.")
    if task == "regression":
        y = pd.to_numeric(y, errors="coerce")
        valid = y.notna()
        X = X.loc[valid]
        y = y.loc[valid]
        if len(y) < 20:
            raise ValueError("Regression requires at least 20 numeric target values.")

    X, _, _, datetime_columns = _prepare_feature_types(X)
    X, dropped_columns = _drop_unusable_columns(X)
    numeric_columns = list(X.select_dtypes(include=np.number).columns)
    categorical_columns = [column for column in X.columns if column not in numeric_columns]
    X, y, sampling_warning = _sample_rows(
        X, y, task, row_limit, random_state
    )

    warnings: list[str] = []
    if sampling_warning:
        warnings.append(sampling_warning)
    if dropped_columns:
        warnings.append(
            "Removed constant or likely identifier columns: "
            + ", ".join(dropped_columns[:12])
            + ("…" if len(dropped_columns) > 12 else "")
        )
    if datetime_columns:
        warnings.append(
            "Expanded date-like columns into year, month, day, and weekday features: "
            + ", ".join(datetime_columns[:8])
        )

    stratify = None
    if task == "classification":
        class_counts = y.value_counts()
        if class_counts.min() >= 2 and len(class_counts) <= max(2, int(len(y) * test_size)):
            stratify = y
        else:
            warnings.append(
                "A stratified split was not possible because at least one class is too small."
            )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    registry = _registry(task)
    unknown = [name for name in model_names if name not in registry]
    if unknown:
        raise ValueError(f"Unknown model(s): {', '.join(unknown)}")

    leaderboard: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    fitted: dict[str, Pipeline] = {}

    for model_name in model_names:
        preprocessor = _build_preprocessor(numeric_columns, categorical_columns)
        estimator = registry[model_name].factory(random_state)
        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        if task == "classification":
            row = {
                "model": model_name,
                "macro_f1": float(
                    f1_score(y_test, predictions, average="macro", zero_division=0)
                ),
                "weighted_f1": float(
                    f1_score(y_test, predictions, average="weighted", zero_division=0)
                ),
                "accuracy": float(accuracy_score(y_test, predictions)),
                "macro_precision": float(
                    precision_score(
                        y_test, predictions, average="macro", zero_division=0
                    )
                ),
                "macro_recall": float(
                    recall_score(
                        y_test, predictions, average="macro", zero_division=0
                    )
                ),
            }
            labels = sorted(pd.unique(pd.concat([y_test, pd.Series(predictions)])), key=str)
            details[model_name] = {
                "summary": registry[model_name].summary,
                "confusion_matrix": confusion_matrix(
                    y_test, predictions, labels=labels
                ).tolist(),
                "class_labels": [str(label) for label in labels],
                "classification_report": classification_report(
                    y_test,
                    predictions,
                    output_dict=True,
                    zero_division=0,
                ),
            }
        else:
            rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
            row = {
                "model": model_name,
                "rmse": rmse,
                "mae": float(mean_absolute_error(y_test, predictions)),
                "r2": float(r2_score(y_test, predictions)),
            }
            details[model_name] = {
                "summary": registry[model_name].summary,
                "confusion_matrix": None,
                "classification_report": None,
            }

        leaderboard.append(row)
        fitted[model_name] = pipeline

    if task == "classification":
        leaderboard.sort(key=lambda row: row["macro_f1"], reverse=True)
        primary_metric = "macro_f1"
    else:
        leaderboard.sort(key=lambda row: row["rmse"])
        primary_metric = "rmse"

    best_model = leaderboard[0]["model"]
    importance = _feature_importance(fitted[best_model])

    return {
        "task": task,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "primary_metric": primary_metric,
        "best_model": best_model,
        "leaderboard": leaderboard,
        "details": details,
        "feature_importance": importance,
        "warnings": warnings,
        "dropped_columns": dropped_columns,
        "datetime_columns": datetime_columns,
    }
