"""Train, tune, evaluate, and save the expense prediction model."""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Any

# Silence Windows Python 3.13 joblib loky wmic warning before importing sklearn/joblib
if "LOKY_MAX_CPU_COUNT" not in os.environ:
    os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

try:
    from .utils import (
        CALIBRATION_PATH,
        DATA_PATH,
        METRICS_PATH,
        MODEL_PATH,
        OUTPUTS_DIR,
        PROCESSED_DATA_PATH,
        configure_logging,
        ensure_directories,
        format_metric,
        save_json,
    )
    from .preprocess import (
        EXPENSE_COLUMNS,
        FEATURE_COLUMNS,
        NUMERICAL_FEATURES,
        TARGET_COLUMN,
        ExpenseRatioRegressor,
        add_target,
        build_behavioral_calibration,
        build_preprocessor,
        detect_outliers_iqr,
        get_feature_target,
        load_data,
        process_and_save_data,
    )
    from .evaluate import (
        regression_metrics,
        save_eda_plots,
        save_feature_importance,
        save_learning_curve,
        save_prediction_plots,
    )
except ImportError:
    from utils import (
        CALIBRATION_PATH,
        DATA_PATH,
        METRICS_PATH,
        MODEL_PATH,
        OUTPUTS_DIR,
        PROCESSED_DATA_PATH,
        configure_logging,
        ensure_directories,
        format_metric,
        save_json,
    )
    from preprocess import (
        EXPENSE_COLUMNS,
        FEATURE_COLUMNS,
        NUMERICAL_FEATURES,
        TARGET_COLUMN,
        ExpenseRatioRegressor,
        add_target,
        build_behavioral_calibration,
        build_preprocessor,
        detect_outliers_iqr,
        get_feature_target,
        load_data,
        process_and_save_data,
    )
    from evaluate import (
        regression_metrics,
        save_eda_plots,
        save_feature_importance,
        save_learning_curve,
        save_prediction_plots,
    )

LOGGER = logging.getLogger(__name__)


def build_models() -> dict[str, Any]:
    """Create candidate regression models comparing ratio ensemble vs additive baselines."""
    models: dict[str, Any] = {
        "HistGradientBoosting Regressor": ExpenseRatioRegressor(
            HistGradientBoostingRegressor(
                max_iter=150,
                max_depth=5,
                min_samples_leaf=20,
                random_state=42,
            )
        ),
        "Gradient Boosting Regressor": ExpenseRatioRegressor(
            GradientBoostingRegressor(
                n_estimators=180,
                max_depth=4,
                learning_rate=0.08,
                subsample=0.85,
                random_state=42,
            )
        ),
        "Random Forest Regressor": ExpenseRatioRegressor(
            RandomForestRegressor(
                n_estimators=120,
                max_depth=12,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1,
            )
        ),
        "Ridge Regression (Ratio)": ExpenseRatioRegressor(Ridge(alpha=10.0)),
        "Linear Regression (Additive Baseline)": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", LinearRegression()),
            ]
        ),
        "Decision Tree (Additive Baseline)": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", DecisionTreeRegressor(random_state=42)),
            ]
        ),
    }

    optional_models = [
        ("XGBoost Regressor", "xgboost", "XGBRegressor"),
        ("LightGBM Regressor", "lightgbm", "LGBMRegressor"),
        ("CatBoost Regressor", "catboost", "CatBoostRegressor"),
    ]
    for display_name, module_name, class_name in optional_models:
        try:
            module = __import__(module_name, fromlist=[class_name])
            model_class = getattr(module, class_name)
            kwargs = {"random_state": 42}
            if display_name == "XGBoost Regressor":
                kwargs.update({"n_estimators": 150, "objective": "reg:squarederror"})
            if display_name == "LightGBM Regressor":
                kwargs.update({"n_estimators": 150, "verbose": -1})
            if display_name == "CatBoost Regressor":
                kwargs.update({"iterations": 150, "verbose": False})
            models[display_name] = ExpenseRatioRegressor(model_class(**kwargs))
        except ImportError:
            LOGGER.debug("%s is not installed. Skipping it.", display_name)

    return models


def train_and_evaluate_models(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, str, Any]:
    """Train all candidate models, evaluate train/test generalization, and select best model."""
    rows: list[dict[str, Any]] = []
    fitted_models: dict[str, Any] = {}

    outlier_mask = x_test["Income"] > 103000

    for name, estimator in build_models().items():
        started_at = time.perf_counter()
        estimator.fit(x_train, y_train)
        training_time = time.perf_counter() - started_at

        # Measure predictions
        predict_started_at = time.perf_counter()
        test_predictions = estimator.predict(x_test)
        prediction_time = time.perf_counter() - predict_started_at

        train_predictions = estimator.predict(x_train)

        # Calculate metrics for train (bias) and test (variance / generalization)
        train_metrics = regression_metrics(y_train, train_predictions)
        test_metrics = regression_metrics(y_test, test_predictions)
        generalization_gap = abs(train_metrics["R2"] - test_metrics["R2"])

        # Check performance on outlier subset
        outlier_mae = float(np.mean(np.abs(y_test[outlier_mask] - test_predictions[outlier_mask])))

        rows.append(
            {
                "Model": name,
                "Train R2": train_metrics["R2"],
                "Test R2": test_metrics["R2"],
                "Generalization Gap": generalization_gap,
                "Test RMSE": test_metrics["RMSE"],
                "Test MAE": test_metrics["MAE"],
                "Accuracy %": test_metrics["Accuracy_Pct"],
                "Within 15%": test_metrics["Within_15Pct_Accuracy"],
                "Outlier MAE": outlier_mae,
                "Training Time (s)": training_time,
                "Prediction Time (s)": prediction_time,
            }
        )
        fitted_models[name] = estimator
        LOGGER.info(
            "%s | Train R2: %.4f | Test R2: %.4f | Gap: %.4f | RMSE: %.2f | Acc: %.2f%% | Outlier MAE: %.2f",
            name,
            train_metrics["R2"],
            test_metrics["R2"],
            generalization_gap,
            test_metrics["RMSE"],
            test_metrics["Accuracy_Pct"],
            outlier_mae,
        )

    comparison = pd.DataFrame(rows).sort_values(
        by=["Test R2", "Test RMSE", "Generalization Gap"],
        ascending=[False, True, True],
    )
    best_name = select_best_model(comparison)
    return comparison, best_name, fitted_models[best_name]


def select_best_model(comparison: pd.DataFrame) -> str:
    """Select the best performing model, preferring robust ensemble architectures over single linear equations."""
    well_fitted = comparison[comparison["Generalization Gap"] <= 0.03]
    if well_fitted.empty:
        well_fitted = comparison

    best_rmse = float(well_fitted["Test RMSE"].min())
    top_tier = well_fitted[well_fitted["Test RMSE"] <= best_rmse * 1.03]

    # Prioritize non-linear ensemble models over 4KB linear equations
    ensemble_priority = [
        "HistGradientBoosting Regressor",
        "Gradient Boosting Regressor",
        "Random Forest Regressor",
    ]
    for model_name in ensemble_priority:
        if model_name in set(top_tier["Model"]):
            LOGGER.info(
                "Selected robust ensemble model %s (RMSE: %.2f) over simple linear models.",
                model_name,
                float(top_tier.loc[top_tier["Model"] == model_name, "Test RMSE"].iloc[0]),
            )
            return model_name

    return str(well_fitted.iloc[0]["Model"])


def tune_best_model(best_model_name: str, model: Any, x_train: pd.DataFrame, y_train: pd.Series) -> Any:
    """Tune hyperparameters using 5-fold cross validation."""
    if isinstance(model, ExpenseRatioRegressor):
        base = model.estimator_
        if isinstance(base, HistGradientBoostingRegressor):
            param_grid = {
                "estimator__max_depth": [4, 5, 6],
                "estimator__min_samples_leaf": [15, 20, 30],
            }
        elif isinstance(base, GradientBoostingRegressor):
            param_grid = {
                "estimator__n_estimators": [150, 200],
                "estimator__max_depth": [3, 4],
                "estimator__learning_rate": [0.06, 0.08],
            }
        elif isinstance(base, RandomForestRegressor):
            param_grid = {
                "estimator__n_estimators": [100, 150],
                "estimator__max_depth": [10, 14],
                "estimator__min_samples_leaf": [2, 4],
            }
        elif isinstance(base, Ridge):
            param_grid = {
                "estimator__alpha": [0.5, 1.0, 10.0, 50.0],
            }
        else:
            LOGGER.info("No tuning grid configured for base of %s. Reusing fitted model.", best_model_name)
            return model
    else:
        LOGGER.info("No tuning grid configured for %s. Reusing fitted model.", best_model_name)
        return model

    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    search.fit(x_train, y_train)
    LOGGER.info("Best CV RMSE: %.4f | Best tuning parameters: %s", -search.best_score_, search.best_params_)
    return search.best_estimator_


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train FinSight expense prediction model.")
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Force re-processing of raw data into data/processed_dataset.csv before training.",
    )
    return parser.parse_args()


def main(force_reprocess: bool = False) -> None:
    """Run the complete end-to-end training and evaluation workflow."""
    configure_logging()
    ensure_directories()

    # Step 1: Process and store data separately if not already stored or forced
    if force_reprocess or not PROCESSED_DATA_PATH.exists():
        LOGGER.info("Processing raw data and saving to %s...", PROCESSED_DATA_PATH)
        process_and_save_data(DATA_PATH, PROCESSED_DATA_PATH)
    else:
        LOGGER.info("Loading existing processed dataset from %s", PROCESSED_DATA_PATH)

    processed_data = load_data(PROCESSED_DATA_PATH)

    # Step 2: Behavioral calibration and EDA plots
    behavioral_calibration = build_behavioral_calibration(processed_data)
    save_json(behavioral_calibration, CALIBRATION_PATH)
    LOGGER.info("Saved behavioral calibration to %s", CALIBRATION_PATH)

    outliers = detect_outliers_iqr(processed_data, NUMERICAL_FEATURES + [TARGET_COLUMN])
    LOGGER.info("IQR outlier counts: %s", outliers)
    save_eda_plots(processed_data, OUTPUTS_DIR)

    # Step 3: Train-test split
    x, y = get_feature_target(processed_data)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )

    # Step 4: Compare models on training and test sets
    comparison, best_model_name, best_model = train_and_evaluate_models(
        x_train,
        x_test,
        y_train,
        y_test,
    )
    comparison.to_csv(OUTPUTS_DIR / "model_comparison.csv", index=False)
    LOGGER.info("Model comparison results:\n%s", comparison.to_string(index=False))

    # Step 5: Hyperparameter tuning with 5-fold Cross-Validation
    tune_start = time.perf_counter()
    tuned_model = tune_best_model(best_model_name, best_model, x_train, y_train)
    tune_time = time.perf_counter() - tune_start

    # Step 6: Final evaluation on test set and train set
    test_start = time.perf_counter()
    test_predictions = tuned_model.predict(x_test)
    predict_time = time.perf_counter() - test_start
    train_predictions = tuned_model.predict(x_train)

    train_metrics = regression_metrics(y_train, train_predictions)
    test_metrics = regression_metrics(y_test, test_predictions)
    generalization_gap = abs(train_metrics["R2"] - test_metrics["R2"])

    outlier_mask = x_test["Income"] > 103000
    outlier_mae = float(np.mean(np.abs(y_test[outlier_mask] - test_predictions[outlier_mask])))
    outlier_rmse = float(np.sqrt(np.mean((y_test[outlier_mask] - test_predictions[outlier_mask]) ** 2)))

    fit_status = (
        "Well-Fitted (No Overfitting or Underfitting)"
        if generalization_gap <= 0.03 and test_metrics["R2"] >= 0.90
        else "Check Generalization Gap"
    )

    # Step 7: Generate diagnostic plots
    save_prediction_plots(y_test, test_predictions, OUTPUTS_DIR)
    save_feature_importance(tuned_model, x_test, y_test, OUTPUTS_DIR)
    save_learning_curve(tuned_model, x, y, OUTPUTS_DIR)

    # Step 8: Persist best model pipeline
    joblib.dump(tuned_model, MODEL_PATH)
    model_size_bytes = MODEL_PATH.stat().st_size
    LOGGER.info("Saved best model (%s, %d bytes) to %s", best_model_name, model_size_bytes, MODEL_PATH)

    # Step 9: Save comprehensive accuracy and fit metrics
    metrics_payload = {
        "Best Model": best_model_name,
        "Fit Status": fit_status,
        "Train R2": format_metric(train_metrics["R2"]),
        "Test R2": format_metric(test_metrics["R2"]),
        "Generalization Gap": format_metric(generalization_gap),
        "Train RMSE": format_metric(train_metrics["RMSE"]),
        "Test RMSE": format_metric(test_metrics["RMSE"]),
        "Accuracy Percentage": format_metric(test_metrics["Accuracy_Pct"]),
        "Within 10% Accuracy": format_metric(test_metrics.get("Within_10Pct_Accuracy", 0.0)),
        "Within 15% Accuracy": format_metric(test_metrics["Within_15Pct_Accuracy"]),
        "MAE": format_metric(test_metrics["MAE"]),
        "MSE": format_metric(test_metrics["MSE"]),
        "RMSE": format_metric(test_metrics["RMSE"]),
        "MAPE": format_metric(test_metrics["MAPE"]),
        "Outlier MAE": format_metric(outlier_mae),
        "Outlier RMSE": format_metric(outlier_rmse),
        "Training Time": format_metric(tune_time),
        "Prediction Time": format_metric(predict_time),
        "Model Size (Bytes)": model_size_bytes,
        "Number of Features": len(x.columns),
        "Dataset Size": len(processed_data),
        "Processed Data Path": str(PROCESSED_DATA_PATH),
        "Outlier Counts": outliers,
    }
    save_json(metrics_payload, METRICS_PATH)
    LOGGER.info("Saved metrics to %s:\n%s", METRICS_PATH, metrics_payload)


if __name__ == "__main__":
    cli_args = parse_args()
    main(force_reprocess=cli_args.reprocess)
