"""Data loading, cleaning, feature engineering, and preprocessing."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

# Silence Windows Python 3.13 joblib loky wmic warning before importing sklearn/joblib
if "LOKY_MAX_CPU_COUNT" not in os.environ:
    os.environ["LOKY_MAX_CPU_COUNT"] = str(max(1, (os.cpu_count() or 4) - 1))

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="joblib")

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

try:
    from .utils import DATA_PATH, PROCESSED_DATA_PATH
except ImportError:
    from utils import DATA_PATH, PROCESSED_DATA_PATH

# Ensure both 'preprocess' and 'src.preprocess' exist in sys.modules for joblib unpickling
if "preprocess" in sys.modules and "src.preprocess" not in sys.modules:
    sys.modules["src.preprocess"] = sys.modules["preprocess"]
elif "src.preprocess" in sys.modules and "preprocess" not in sys.modules:
    sys.modules["preprocess"] = sys.modules["src.preprocess"]

LOGGER = logging.getLogger(__name__)

FEATURE_COLUMNS = ["Income", "Age", "Dependents", "Occupation", "City_Tier"]
NUMERICAL_FEATURES = ["Income", "Age", "Dependents"]
CATEGORICAL_FEATURES = ["Occupation", "City_Tier"]
EXPENSE_COLUMNS = ["Rent", "Loan_Repayment", "Insurance", "Groceries", "Transport"]
TARGET_COLUMN = "Total_Expense"
RATIO_COLUMN = "Expense_Ratio"
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
    """Load dataset from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    data = pd.read_csv(path)
    LOGGER.info("Loaded dataset from %s | shape: %s", path, data.shape)
    return data


def validate_required_columns(data: pd.DataFrame) -> None:
    """Validate that all columns required by the pipeline exist."""
    required_columns = set(FEATURE_COLUMNS + EXPENSE_COLUMNS)
    missing_columns = sorted(required_columns.difference(data.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Clean duplicate rows, impossible values, and normalize categorical labels."""
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
    """Normalize user-facing occupation inputs to standard training categories."""
    if pd.isna(value):
        return pd.NA
    cleaned = str(value).strip()
    if not cleaned:
        return pd.NA
    return OCCUPATION_ALIASES.get(cleaned.lower(), cleaned)


def normalize_city_tier(value: object) -> object:
    """Normalize city tier inputs to standard training categories."""
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
    """Fail fast when prediction categories are unknown."""
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
    """Compute Total_Expense and bounded Expense_Ratio without leakage."""
    engineered = data.copy()
    engineered[TARGET_COLUMN] = engineered[EXPENSE_COLUMNS].sum(
        axis=1,
        min_count=len(EXPENSE_COLUMNS),
    )
    income_series = pd.to_numeric(engineered["Income"], errors="coerce")
    positive_incomes = income_series[income_series > 0]
    fallback_median = float(positive_incomes.median()) if not positive_incomes.empty else 30000.0
    safe_income = income_series.fillna(fallback_median)
    safe_income = safe_income.where(safe_income > 0, fallback_median).clip(lower=1.0)
    raw_ratio = engineered[TARGET_COLUMN] / safe_income
    engineered[RATIO_COLUMN] = raw_ratio.clip(lower=0.05, upper=0.95)
    return engineered


def get_feature_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return leakage-free feature matrix and target vector."""
    if TARGET_COLUMN not in data.columns:
        data = add_target(data)
    model_data = data[FEATURE_COLUMNS + [TARGET_COLUMN]].dropna(subset=[TARGET_COLUMN])
    return model_data[FEATURE_COLUMNS], model_data[TARGET_COLUMN]


def build_preprocessor() -> ColumnTransformer:
    """Build a reusable preprocessing transformer using RobustScaler for outlier resilience."""
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
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


def build_behavioral_calibration(data: pd.DataFrame) -> dict[str, Any]:
    """Build small empirical adjustments for weak but user-visible features."""
    if TARGET_COLUMN not in data.columns:
        calibration_data = add_target(data).dropna(subset=[TARGET_COLUMN])
    else:
        calibration_data = data.dropna(subset=[TARGET_COLUMN])

    overall_mean = float(calibration_data[TARGET_COLUMN].mean())
    dependent_means = calibration_data.groupby("Dependents")[TARGET_COLUMN].mean()
    dependent_x = dependent_means.index.astype(float).to_numpy()
    dependent_y = dependent_means.to_numpy()
    dependent_slope = 0.0
    if len(dependent_x) > 1:
        dependent_slope = float(np.polyfit(dependent_x, dependent_y, deg=1)[0])

    age_bins = [17, 25, 35, 45, 55, 100]
    age_labels = ["18-25", "26-35", "36-45", "46-55", "56+"]
    age_band = pd.cut(
        calibration_data["Age"],
        bins=age_bins,
        labels=age_labels,
        include_lowest=True,
    )
    age_offsets = (
        calibration_data.groupby(age_band, observed=True)[TARGET_COLUMN].mean()
        - overall_mean
    )

    return {
        "dependent_baseline": float(calibration_data["Dependents"].median()),
        "dependent_effect_per_person": round(dependent_slope, 4),
        "age_band_offsets": {
            str(age_band_name): round(float(offset), 4)
            for age_band_name, offset in age_offsets.items()
        },
    }


def process_and_save_data(
    input_path: Path = DATA_PATH,
    output_path: Path = PROCESSED_DATA_PATH,
) -> pd.DataFrame:
    """Clean raw data, compute target & ratios, and persist processed data separately."""
    LOGGER.info("Starting data processing from %s", input_path)
    raw_data = load_data(input_path)
    cleaned = clean_data(raw_data)
    processed = add_target(cleaned)

    # Validate output
    assert TARGET_COLUMN in processed.columns, f"{TARGET_COLUMN} must be in processed data"
    assert RATIO_COLUMN in processed.columns, f"{RATIO_COLUMN} must be in processed data"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(output_path, index=False)
    LOGGER.info(
        "Successfully saved processed dataset (%d rows, %d columns) to %s",
        len(processed),
        len(processed.columns),
        output_path,
    )
    return processed


class ExpenseRatioRegressor(BaseEstimator, RegressorMixin):
    """Predicts expense by modeling the income spending ratio.

    This architecture resolves outlier vulnerabilities:
    1. Low Income: Bounded ratio prevents expense from exceeding income.
    2. High Income: Proportional scaling ensures realistic expenses reflecting city tier differences.
    3. Model capacity: Uses ensemble tree models to capture non-linear demographic interactions,
       avoiding underfitting without overfitting.
    """

    __module__ = "src.preprocess"

    def __init__(
        self,
        estimator: Any = None,
        min_ratio: float = 0.05,
        max_ratio: float = 0.95,
    ) -> None:
        self.estimator = estimator
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> ExpenseRatioRegressor:
        if isinstance(X, pd.DataFrame):
            X_df = X[FEATURE_COLUMNS].copy() if all(c in X.columns for c in FEATURE_COLUMNS) else X.copy()
        else:
            X_df = pd.DataFrame(X, columns=FEATURE_COLUMNS)

        y_ser = pd.Series(y) if not isinstance(y, pd.Series) else y.copy()

        self.preprocessor_ = build_preprocessor()
        X_trans = self.preprocessor_.fit_transform(X_df)

        imputer = self.preprocessor_.named_transformers_["num"].named_steps["imputer"]
        self.income_median_ = float(imputer.statistics_[0])

        income_arr = pd.to_numeric(X_df["Income"], errors="coerce").to_numpy(dtype=float)
        safe_income = np.where(np.isnan(income_arr) | (income_arr <= 0), self.income_median_, income_arr)
        safe_income = np.maximum(1.0, safe_income)

        y_arr = y_ser.to_numpy(dtype=float)
        valid_y = y_arr[~np.isnan(y_arr)]
        fallback_y = float(np.median(valid_y)) if len(valid_y) > 0 else self.income_median_ * 0.5
        y_safe = np.where(np.isnan(y_arr), fallback_y, y_arr)

        target_ratio = np.clip(
            y_safe / safe_income,
            self.min_ratio,
            self.max_ratio,
        )

        base = self.estimator if self.estimator is not None else HistGradientBoostingRegressor(
            max_iter=150, max_depth=5, min_samples_leaf=20, random_state=42
        )
        self.estimator_ = clone(base)
        self.estimator_.fit(X_trans, target_ratio)

        # Expose named_steps for evaluation helpers and pipeline compatibility
        self.named_steps = {
            "preprocessor": self.preprocessor_,
            "model": self.estimator_,
        }
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X_df = X[FEATURE_COLUMNS].copy() if all(c in X.columns for c in FEATURE_COLUMNS) else X.copy()
        else:
            X_df = pd.DataFrame(X, columns=FEATURE_COLUMNS)

        X_trans = self.preprocessor_.transform(X_df)
        pred_ratio = self.estimator_.predict(X_trans)
        pred_ratio = np.clip(pred_ratio, self.min_ratio, self.max_ratio)

        income_raw = pd.to_numeric(X_df["Income"], errors="coerce").to_numpy(dtype=float)
        fallback_income = getattr(self, "income_median_", 30000.0)
        safe_income = np.where(np.isnan(income_raw) | (income_raw < 0), fallback_income, income_raw)
        return np.round(pred_ratio * safe_income, 2)


if __name__ == "__main__":
    from utils import configure_logging
    configure_logging()
    process_and_save_data()
