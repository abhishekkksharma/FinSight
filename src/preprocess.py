"""Data loading, cleaning, feature engineering, and preprocessing."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


LOGGER = logging.getLogger(__name__)

FEATURE_COLUMNS = ["Income", "Age", "Dependents", "Occupation", "City_Tier"]
NUMERICAL_FEATURES = ["Income", "Age", "Dependents"]
CATEGORICAL_FEATURES = ["Occupation", "City_Tier"]
EXPENSE_COLUMNS = ["Rent", "Loan_Repayment", "Insurance", "Groceries", "Transport"]
TARGET_COLUMN = "Total_Expense"
VALID_OCCUPATIONS = {"Professional", "Retired", "Self_Employed", "Student"}
VALID_CITY_TIERS = {"Tier_1", "Tier_2", "Tier_3"}

OCCUPATION_ALIASES = {
    "engineer": "Professional",
    "software engineer": "Professional",
    "developer": "Professional",
    "doctor": "Professional",
    "teacher": "Professional",
    "professional": "Professional",
    "self employed": "Self_Employed",
    "self-employed": "Self_Employed",
    "self_employed": "Self_Employed",
    "business": "Self_Employed",
    "student": "Student",
    "retired": "Retired",
}
CITY_TIER_ALIASES = {
    "tier 1": "Tier_1",
    "tier-1": "Tier_1",
    "tier_1": "Tier_1",
    "1": "Tier_1",
    "tier 2": "Tier_2",
    "tier-2": "Tier_2",
    "tier_2": "Tier_2",
    "2": "Tier_2",
    "tier 3": "Tier_3",
    "tier-3": "Tier_3",
    "tier_3": "Tier_3",
    "3": "Tier_3",
}


def load_data(path: Path) -> pd.DataFrame:
    """Load the source CSV dataset."""
    data = pd.read_csv(path)
    LOGGER.info("Dataset shape: %s", data.shape)
    LOGGER.info("Dataset columns: %s", list(data.columns))
    LOGGER.info("First five rows:\n%s", data.head().to_string())
    LOGGER.info("Missing values:\n%s", data.isna().sum().to_string())
    LOGGER.info("Data types:\n%s", data.dtypes.to_string())
    return data


def validate_required_columns(data: pd.DataFrame) -> None:
    """Validate that all columns required by the training pipeline exist."""
    required_columns = set(FEATURE_COLUMNS + EXPENSE_COLUMNS)
    missing_columns = sorted(required_columns.difference(data.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Clean duplicate rows, impossible values, and basic text inconsistencies."""
    validate_required_columns(data)
    cleaned = data.copy()
    before = len(cleaned)
    cleaned = cleaned.drop_duplicates()
    LOGGER.info("Removed %s duplicate rows.", before - len(cleaned))

    for column in NUMERICAL_FEATURES + EXPENSE_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    impossible_masks = {
        "Income": cleaned["Income"] < 0,
        "Age": (cleaned["Age"] < 0) | (cleaned["Age"] > 100),
        "Dependents": cleaned["Dependents"] < 0,
    }
    for column in EXPENSE_COLUMNS:
        impossible_masks[column] = cleaned[column] < 0

    for column, mask in impossible_masks.items():
        count = int(mask.fillna(False).sum())
        if count:
            LOGGER.warning("Setting %s impossible values in %s to missing.", count, column)
            cleaned.loc[mask, column] = pd.NA

    cleaned["Occupation"] = cleaned["Occupation"].apply(normalize_occupation)
    cleaned["City_Tier"] = cleaned["City_Tier"].apply(normalize_city_tier)

    return cleaned


def normalize_occupation(value: object) -> object:
    """Normalize user-facing occupation inputs to training categories."""
    if pd.isna(value):
        return pd.NA
    cleaned = str(value).strip()
    if not cleaned:
        return pd.NA
    return OCCUPATION_ALIASES.get(cleaned.lower(), cleaned)


def normalize_city_tier(value: object) -> object:
    """Normalize city tier inputs to training categories."""
    if pd.isna(value):
        return pd.NA
    cleaned = str(value).strip()
    if not cleaned:
        return pd.NA
    return CITY_TIER_ALIASES.get(cleaned.lower(), cleaned)


def normalize_prediction_input(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw prediction payloads before model inference."""
    normalized = data.copy()
    normalized["Occupation"] = normalized["Occupation"].apply(normalize_occupation)
    normalized["City_Tier"] = normalized["City_Tier"].apply(normalize_city_tier)
    validate_prediction_categories(normalized)
    return normalized


def validate_prediction_categories(data: pd.DataFrame) -> None:
    """Fail fast when prediction categories were not seen during training."""
    occupations = set(data["Occupation"].dropna().astype(str))
    city_tiers = set(data["City_Tier"].dropna().astype(str))
    unknown_occupations = sorted(occupations.difference(VALID_OCCUPATIONS))
    unknown_city_tiers = sorted(city_tiers.difference(VALID_CITY_TIERS))
    if unknown_occupations or unknown_city_tiers:
        raise ValueError(
            "Unknown prediction categories. "
            f"Unknown occupations: {unknown_occupations}. "
            f"Use one of {sorted(VALID_OCCUPATIONS)} or a supported alias. "
            f"Unknown city tiers: {unknown_city_tiers}. "
            f"Use one of {sorted(VALID_CITY_TIERS)}."
        )


def add_target(data: pd.DataFrame) -> pd.DataFrame:
    """Create Total_Expense from the approved expense columns."""
    engineered = data.copy()
    engineered[TARGET_COLUMN] = engineered[EXPENSE_COLUMNS].sum(
        axis=1,
        min_count=len(EXPENSE_COLUMNS),
    )
    return engineered


def get_feature_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return leakage-free feature matrix and target vector."""
    engineered = add_target(data)
    model_data = engineered[FEATURE_COLUMNS + [TARGET_COLUMN]].dropna(subset=[TARGET_COLUMN])
    return model_data[FEATURE_COLUMNS], model_data[TARGET_COLUMN]


def build_preprocessor() -> ColumnTransformer:
    """Build a reusable preprocessing transformer."""
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", encoder),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, NUMERICAL_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def detect_outliers_iqr(data: pd.DataFrame, columns: list[str]) -> dict[str, int]:
    """Count IQR outliers in selected numeric columns."""
    outliers: dict[str, int] = {}
    for column in columns:
        series = pd.to_numeric(data[column], errors="coerce").dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers[column] = int(((series < lower) | (series > upper)).sum())
    return outliers
