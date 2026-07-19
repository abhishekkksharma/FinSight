"""Model evaluation and visualization helpers."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import learning_curve

from preprocess import TARGET_COLUMN


LOGGER = logging.getLogger(__name__)
sns.set_theme(style="whitegrid")


def regression_metrics(y_true: pd.Series, y_pred: pd.Series | list[float]) -> dict[str, float]:
    """Calculate core regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": mse**0.5,
        "R2": r2_score(y_true, y_pred),
    }


def save_eda_plots(data: pd.DataFrame, output_dir: Path) -> None:
    """Generate and save exploratory data analysis plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_specs = [
        ("Income", "hist", "income_distribution.png"),
        ("Age", "hist", "age_distribution.png"),
        ("Occupation", "count", "occupation_frequency.png"),
        ("City_Tier", "count", "city_tier_frequency.png"),
        ("Dependents", "count", "dependents_distribution.png"),
        (TARGET_COLUMN, "hist", "target_distribution.png"),
    ]

    for column, plot_type, filename in plot_specs:
        plt.figure(figsize=(9, 5))
        if plot_type == "hist":
            sns.histplot(data[column], kde=True, bins=35)
        else:
            sns.countplot(data=data, x=column, order=data[column].value_counts().index)
            plt.xticks(rotation=30, ha="right")
        plt.title(column.replace("_", " "))
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=150)
        plt.close()

    numeric_data = data.select_dtypes(include=["number"])
    plt.figure(figsize=(12, 9))
    sns.heatmap(numeric_data.corr(), cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=150)
    plt.close()

    boxplot_columns = ["Income", "Age", "Dependents", TARGET_COLUMN]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, column in zip(axes.flatten(), boxplot_columns):
        sns.boxplot(y=data[column], ax=axis)
        axis.set_title(column.replace("_", " "))
    plt.tight_layout()
    plt.savefig(output_dir / "boxplots_numerical.png", dpi=150)
    plt.close()
    LOGGER.info("EDA plots saved to %s", output_dir)


def save_prediction_plots(
    y_true: pd.Series,
    y_pred: pd.Series | list[float],
    output_dir: Path,
) -> None:
    """Save prediction-vs-actual and residual plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    residuals = y_true - y_pred

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.65)
    lower = min(y_true.min(), min(y_pred))
    upper = max(y_true.max(), max(y_pred))
    plt.plot([lower, upper], [lower, upper], color="red", linestyle="--")
    plt.xlabel("Actual Monthly Expense")
    plt.ylabel("Predicted Monthly Expense")
    plt.title("Prediction vs Actual")
    plt.tight_layout()
    plt.savefig(output_dir / "prediction_vs_actual.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.65)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted Monthly Expense")
    plt.ylabel("Residual")
    plt.title("Residual Plot")
    plt.tight_layout()
    plt.savefig(output_dir / "residual_plot.png", dpi=150)
    plt.close()


def save_feature_importance(
    model,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path,
) -> None:
    """Save model feature importance, falling back to permutation importance."""
    output_dir.mkdir(parents=True, exist_ok=True)
    estimator = model.named_steps["model"]
    preprocessor = model.named_steps["preprocessor"]
    feature_names = list(preprocessor.get_feature_names_out())

    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importances = abs(estimator.coef_)
    else:
        result = permutation_importance(
            model,
            x_test,
            y_test,
            n_repeats=5,
            random_state=42,
            scoring="neg_root_mean_squared_error",
        )
        feature_names = list(x_test.columns)
        importances = result.importances_mean

    importance_data = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(20)
    )

    plt.figure(figsize=(10, 7))
    sns.barplot(data=importance_data, x="importance", y="feature")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png", dpi=150)
    plt.close()


def save_learning_curve(model, x: pd.DataFrame, y: pd.Series, output_dir: Path) -> None:
    """Save a learning curve using RMSE as the score."""
    train_sizes, train_scores, validation_scores = learning_curve(
        model,
        x,
        y,
        cv=5,
        scoring="neg_root_mean_squared_error",
        train_sizes=[0.1, 0.325, 0.55, 0.775, 1.0],
        n_jobs=-1,
    )
    train_rmse = -train_scores.mean(axis=1)
    validation_rmse = -validation_scores.mean(axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_rmse, marker="o", label="Training RMSE")
    plt.plot(train_sizes, validation_rmse, marker="o", label="Validation RMSE")
    plt.xlabel("Training Examples")
    plt.ylabel("RMSE")
    plt.title("Learning Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "learning_curve.png", dpi=150)
    plt.close()
