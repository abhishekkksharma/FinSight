"""Train, tune, evaluate, and save the expense prediction model."""

from __future__ import annotations

import logging
import time
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from evaluate import (
    regression_metrics,
    save_eda_plots,
    save_feature_importance,
    save_learning_curve,
    save_prediction_plots,
)
from preprocess import (
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
    add_target,
    build_preprocessor,
    clean_data,
    detect_outliers_iqr,
    get_feature_target,
    load_data,
)
from utils import (
    DATA_PATH,
    METRICS_PATH,
    MODEL_PATH,
    OUTPUTS_DIR,
    configure_logging,
    ensure_directories,
    format_metric,
    save_json,
)


LOGGER = logging.getLogger(__name__)


def build_models() -> dict[str, Any]:
    """Create candidate regression models."""
    models: dict[str, Any] = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=160,
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42),
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
                kwargs.update({"n_estimators": 200, "objective": "reg:squarederror"})
            if display_name == "LightGBM Regressor":
                kwargs.update({"n_estimators": 200, "verbose": -1})
            if display_name == "CatBoost Regressor":
                kwargs.update({"iterations": 200, "verbose": False})
            models[display_name] = model_class(**kwargs)
        except ImportError:
            LOGGER.info("%s is not installed. Skipping it.", display_name)

    return models


def train_and_evaluate_models(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, str, Pipeline]:
    """Train all candidate models and return the best fitted pipeline."""
    rows: list[dict[str, Any]] = []
    fitted_models: dict[str, Pipeline] = {}

    for name, estimator in build_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", estimator),
            ]
        )
        started_at = time.perf_counter()
        pipeline.fit(x_train, y_train)
        training_time = time.perf_counter() - started_at

        predict_started_at = time.perf_counter()
        predictions = pipeline.predict(x_test)
        prediction_time = time.perf_counter() - predict_started_at

        metrics = regression_metrics(y_test, predictions)
        rows.append(
            {
                "Model": name,
                **metrics,
                "Training Time": training_time,
                "Prediction Time": prediction_time,
            }
        )
        fitted_models[name] = pipeline
        LOGGER.info(
            "%s | RMSE: %.4f | R2: %.4f",
            name,
            metrics["RMSE"],
            metrics["R2"],
        )

    comparison = pd.DataFrame(rows).sort_values(
        by=["R2", "RMSE"],
        ascending=[False, True],
    )
    best_name = str(comparison.iloc[0]["Model"])
    return comparison, best_name, fitted_models[best_name]


def tune_best_model(best_model_name: str, pipeline: Pipeline, x_train, y_train) -> Pipeline:
    """Tune the best model family with 5-fold cross-validation."""
    model = pipeline.named_steps["model"]
    if isinstance(model, RandomForestRegressor):
        param_grid = {
            "model__n_estimators": [120, 200],
            "model__max_depth": [None, 12, 20],
            "model__min_samples_split": [2, 5],
        }
    elif isinstance(model, GradientBoostingRegressor):
        param_grid = {
            "model__n_estimators": [120, 200],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [2, 3],
        }
    elif isinstance(model, DecisionTreeRegressor):
        param_grid = {
            "model__max_depth": [None, 8, 12, 20],
            "model__min_samples_split": [2, 5, 10],
        }
    else:
        LOGGER.info("No tuning grid configured for %s. Reusing fitted model.", best_model_name)
        return pipeline

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    search.fit(x_train, y_train)
    LOGGER.info("Best tuning parameters: %s", search.best_params_)
    return search.best_estimator_


def main() -> None:
    """Run the full model training workflow."""
    configure_logging()
    ensure_directories()

    raw_data = load_data(DATA_PATH)
    cleaned_data = clean_data(raw_data)
    model_data = add_target(cleaned_data)
    outliers = detect_outliers_iqr(model_data, NUMERICAL_FEATURES + [TARGET_COLUMN])
    LOGGER.info("IQR outlier counts: %s", outliers)
    save_eda_plots(model_data, OUTPUTS_DIR)

    x, y = get_feature_target(cleaned_data)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )

    comparison, best_model_name, best_pipeline = train_and_evaluate_models(
        x_train,
        x_test,
        y_train,
        y_test,
    )
    comparison.to_csv(OUTPUTS_DIR / "model_comparison.csv", index=False)
    LOGGER.info("Model comparison:\n%s", comparison.to_string(index=False))

    started_at = time.perf_counter()
    tuned_pipeline = tune_best_model(best_model_name, best_pipeline, x_train, y_train)
    tuning_training_time = time.perf_counter() - started_at

    predict_started_at = time.perf_counter()
    tuned_predictions = tuned_pipeline.predict(x_test)
    prediction_time = time.perf_counter() - predict_started_at
    final_metrics = regression_metrics(y_test, tuned_predictions)

    save_prediction_plots(y_test, tuned_predictions, OUTPUTS_DIR)
    save_feature_importance(tuned_pipeline, x_test, y_test, OUTPUTS_DIR)
    save_learning_curve(tuned_pipeline, x, y, OUTPUTS_DIR)

    joblib.dump(tuned_pipeline, MODEL_PATH)
    LOGGER.info("Saved best model pipeline to %s", MODEL_PATH)

    metrics_payload = {
        "Best Model": best_model_name,
        "MAE": format_metric(final_metrics["MAE"]),
        "MSE": format_metric(final_metrics["MSE"]),
        "RMSE": format_metric(final_metrics["RMSE"]),
        "R2": format_metric(final_metrics["R2"]),
        "Training Time": format_metric(tuning_training_time),
        "Prediction Time": format_metric(prediction_time),
        "Number of Features": len(x.columns),
        "Dataset Size": len(model_data),
        "Outlier Counts": outliers,
    }
    save_json(metrics_payload, METRICS_PATH)
    LOGGER.info("Saved metrics to %s", METRICS_PATH)


if __name__ == "__main__":
    main()
