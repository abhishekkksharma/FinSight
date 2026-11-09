"""Comprehensive tests for data preprocessing, model prediction, outlier robustness, and affordability."""

from __future__ import annotations

import os
import sys

# Silence Windows Python 3.13 joblib loky wmic warning before joblib/sklearn loads
if "LOKY_MAX_CPU_COUNT" not in os.environ:
    os.environ["LOKY_MAX_CPU_COUNT"] = str(max(1, (os.cpu_count() or 4) - 1))

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="joblib")

import json
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.utils import DATA_PATH, PROCESSED_DATA_PATH, MODEL_PATH, METRICS_PATH
from src.preprocess import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    clean_data,
    add_target,
    load_data,
    get_feature_target,
    process_and_save_data,
    normalize_occupation,
    normalize_city_tier,
    ExpenseRatioRegressor,
)
from src.predict import predict_expense, assess_affordability, load_model
from src.evaluate import regression_metrics
from src.api import predict, health, PredictionRequest


class TestDataProcessing(unittest.TestCase):
    """Test data cleaning, transformation, and processed storage."""

    def test_clean_data_and_aliases(self):
        df_raw = pd.DataFrame({
            "Income": [50000.0, -100.0, 75000.0],
            "Age": [30, 150, 40],
            "Dependents": [2, -1, 0],
            "Occupation": ["Software Engineer", "Doctor", "student"],
            "City_Tier": ["tier-1", "Tier 2", "3"],
            "Rent": [15000.0, 10000.0, 11250.0],
            "Loan_Repayment": [0.0, 0.0, 0.0],
            "Insurance": [2500.0, 1000.0, 2000.0],
            "Groceries": [6000.0, 4000.0, 7500.0],
            "Transport": [3000.0, 2000.0, 4000.0],
        })
        cleaned = clean_data(df_raw)
        self.assertEqual(len(cleaned), 3)
        self.assertTrue(pd.isna(cleaned.loc[1, "Income"]))
        self.assertTrue(pd.isna(cleaned.loc[1, "Age"]))
        self.assertTrue(pd.isna(cleaned.loc[1, "Dependents"]))
        self.assertEqual(cleaned.loc[0, "Occupation"], "Professional")
        self.assertEqual(cleaned.loc[2, "Occupation"], "Student")
        self.assertEqual(cleaned.loc[0, "City_Tier"], "Tier_1")
        self.assertEqual(cleaned.loc[2, "City_Tier"], "Tier_3")

    def test_processed_data_file_exists_and_valid(self):
        self.assertTrue(PROCESSED_DATA_PATH.exists(), "Processed data file must exist on disk.")
        processed = pd.read_csv(PROCESSED_DATA_PATH)
        self.assertIn(TARGET_COLUMN, processed.columns)
        self.assertIn("Expense_Ratio", processed.columns)
        self.assertGreater(len(processed), 1000)
        for col in FEATURE_COLUMNS:
            self.assertIn(col, processed.columns)

    def test_fit_with_impossible_or_nan_income_succeeds(self):
        """Verify ExpenseRatioRegressor.fit handles missing/invalid income without NaN crash."""
        raw = pd.DataFrame({
            "Income": [-100.0, 50000.0, 60000.0, 70000.0, 80000.0],
            "Age": [30, 40, 50, 28, 35],
            "Dependents": [1, 2, 0, 1, 3],
            "Occupation": ["Professional", "Student", "Retired", "Professional", "Self_Employed"],
            "City_Tier": ["Tier_1", "Tier_2", "Tier_3", "Tier_1", "Tier_2"],
            "Rent": [5000, 15000, 18000, 20000, 22000],
            "Loan_Repayment": [0, 0, 0, 0, 0],
            "Insurance": [1000, 2000, 2500, 2500, 3000],
            "Groceries": [3000, 6000, 7000, 7500, 8000],
            "Transport": [1000, 3000, 3500, 4000, 4500],
        })
        cleaned = clean_data(raw)
        processed = add_target(cleaned)
        x, y = get_feature_target(processed)
        model = ExpenseRatioRegressor()
        # Must fit cleanly without 'ValueError: Input y contains NaN'
        model.fit(x, y)
        preds = model.predict(x)
        self.assertEqual(len(preds), 5)
        self.assertTrue(np.all(np.isfinite(preds)))


class TestOutlierRobustness(unittest.TestCase):
    """Verify that the model performs smoothly and safely on extreme outlying data."""

    def test_low_income_outlier(self):
        pred = predict_expense(income=3000, age=30, dependents=1, occupation="Professional", city_tier="Tier_1")
        self.assertGreater(pred, 0)
        self.assertLess(pred, 3000, "Low income expense must not exceed income.")

    def test_high_income_outlier(self):
        pred = predict_expense(income=1000000, age=40, dependents=2, occupation="Professional", city_tier="Tier_1")
        ratio = pred / 1000000
        self.assertGreater(ratio, 0.45, "High income expense ratio should reflect real city living expenses.")
        self.assertLess(ratio, 0.75, "High income expense ratio should remain bounded.")

    def test_model_size_not_small(self):
        size = MODEL_PATH.stat().st_size
        self.assertGreater(size, 20000, f"Model size is too small ({size} bytes). Should be substantial ensemble.")

    def test_predict_nan_income_batch_fallback(self):
        """Model predict should use imputed median income instead of outputting NaN when income is missing."""
        model = load_model()
        df = pd.DataFrame([{
            "Income": np.nan,
            "Age": 30,
            "Dependents": 1,
            "Occupation": "Professional",
            "City_Tier": "Tier_1"
        }])
        pred = model.predict(df)[0]
        self.assertTrue(np.isfinite(pred), "Prediction with missing income should produce a finite number.")
        self.assertGreater(pred, 0)

    def test_invalid_income_validation(self):
        """predict_expense must reject income <= 0 with ValueError."""
        with self.assertRaises(ValueError):
            predict_expense(income=0, age=30, dependents=1, occupation="Professional", city_tier="Tier_1")
        with self.assertRaises(ValueError):
            predict_expense(income=-5000, age=30, dependents=1, occupation="Professional", city_tier="Tier_1")


class TestEvaluationAndMetrics(unittest.TestCase):
    """Test accuracy calculations and fit metrics."""

    def test_regression_metrics(self):
        y_true = np.array([10000.0, 20000.0, 30000.0])
        y_pred = np.array([10500.0, 19500.0, 30500.0])
        metrics = regression_metrics(y_true, y_pred)
        self.assertIn("Accuracy_Pct", metrics)
        self.assertGreater(metrics["Accuracy_Pct"], 90.0)
        self.assertIn("Within_15Pct_Accuracy", metrics)
        self.assertEqual(metrics["Within_15Pct_Accuracy"], 100.0)
        self.assertIn("Within_10Pct_Accuracy", metrics)
        self.assertEqual(metrics["Within_10Pct_Accuracy"], 100.0)
        self.assertGreater(metrics["R2"], 0.95)

    def test_metrics_json_fit_quality(self):
        """Verify saved metrics indicate good fit without underfitting or overfitting."""
        self.assertTrue(METRICS_PATH.exists(), "metrics.json must exist.")
        with METRICS_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("Fit Status", data)
        self.assertEqual(data["Fit Status"], "Well-Fitted (No Overfitting or Underfitting)")
        self.assertLessEqual(data["Generalization Gap"], 0.03)
        self.assertGreaterEqual(data["Test R2"], 0.90)
        self.assertGreaterEqual(data["Accuracy Percentage"], 80.0)


class TestAffordability(unittest.TestCase):
    """Test EMI affordability decision logic."""

    def test_affordable(self):
        result = assess_affordability(income=100000, predicted_expense=40000, monthly_emi=10000, min_savings_rate=0.20)
        self.assertEqual(result["verdict"], "Affordable")
        self.assertEqual(result["surplus_after_emi"], 50000)

    def test_tight(self):
        result = assess_affordability(income=100000, predicted_expense=60000, monthly_emi=25000, min_savings_rate=0.20)
        self.assertEqual(result["verdict"], "Tight")

    def test_not_affordable(self):
        result = assess_affordability(income=50000, predicted_expense=40000, monthly_emi=15000, min_savings_rate=0.20)
        self.assertEqual(result["verdict"], "Not Affordable")


class TestAPIEndpoints(unittest.TestCase):
    """Test FastAPI endpoint handlers."""

    def test_health_and_predict(self):
        h = health()
        self.assertEqual(h["status"], "ok")

        req = PredictionRequest(
            income=80000,
            age=28,
            dependents=1,
            occupation="Engineer",
            city_tier="Tier-1",
            monthly_emi=10000,
            tenure_years=3,
            min_savings_rate=0.20,
        )
        res = predict(req)
        self.assertGreater(res.predicted_monthly_expense, 0)
        self.assertIn("verdict", res.affordability)


if __name__ == "__main__":
    unittest.main()
