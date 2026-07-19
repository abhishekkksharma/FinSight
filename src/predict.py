"""Prediction interface for the trained expense model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from preprocess import FEATURE_COLUMNS, normalize_prediction_input
from utils import MODEL_PATH


def load_model(model_path: Path = MODEL_PATH):
    """Load the fitted model pipeline."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run `python src/train.py` first."
        )
    return joblib.load(model_path)


def predict_expense(
    income: float,
    age: int,
    dependents: int,
    occupation: str,
    city_tier: str,
    model_path: Path = MODEL_PATH,
) -> float:
    """Predict a user's expected monthly expense."""
    model = load_model(model_path)
    payload = pd.DataFrame(
        [
            {
                "Income": income,
                "Age": age,
                "Dependents": dependents,
                "Occupation": occupation,
                "City_Tier": city_tier,
            }
        ],
        columns=FEATURE_COLUMNS,
    )
    payload = normalize_prediction_input(payload)
    prediction = model.predict(payload)[0]
    return round(float(prediction), 2)


def assess_affordability(
    income: float,
    predicted_expense: float,
    monthly_emi: float,
    tenure_years: int = 5,
    min_savings_rate: float = 0.20,
) -> dict[str, Any]:
    """Assess affordability for an additional monthly EMI."""
    if income <= 0:
        raise ValueError("Income must be greater than zero.")
    if monthly_emi < 0:
        raise ValueError("Monthly EMI cannot be negative.")
    if tenure_years <= 0:
        raise ValueError("Tenure must be at least one year.")
    if not 0 <= min_savings_rate < 1:
        raise ValueError("Minimum savings rate must be between 0 and 1.")

    current_surplus = income - predicted_expense
    surplus_after_emi = current_surplus - monthly_emi
    required_monthly_buffer = income * min_savings_rate
    emi_to_income_ratio = monthly_emi / income
    total_emi_commitment = monthly_emi * tenure_years * 12
    is_affordable = surplus_after_emi >= required_monthly_buffer

    if is_affordable:
        verdict = "Affordable"
        reason = "Projected surplus after EMI meets the selected savings buffer."
    elif surplus_after_emi >= 0:
        verdict = "Tight"
        reason = "EMI is payable, but it leaves less than the selected savings buffer."
    else:
        verdict = "Not Affordable"
        reason = "Predicted expenses plus EMI exceed monthly income."

    return {
        "verdict": verdict,
        "reason": reason,
        "income": round(float(income), 2),
        "predicted_monthly_expense": round(float(predicted_expense), 2),
        "monthly_emi": round(float(monthly_emi), 2),
        "tenure_years": int(tenure_years),
        "total_emi_commitment": round(float(total_emi_commitment), 2),
        "current_surplus": round(float(current_surplus), 2),
        "surplus_after_emi": round(float(surplus_after_emi), 2),
        "required_monthly_buffer": round(float(required_monthly_buffer), 2),
        "emi_to_income_ratio": round(float(emi_to_income_ratio), 4),
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Predict monthly expense.")
    parser.add_argument("--income", type=float, required=True)
    parser.add_argument("--age", type=int, required=True)
    parser.add_argument("--dependents", type=int, required=True)
    parser.add_argument("--occupation", type=str, required=True)
    parser.add_argument("--city-tier", type=str, required=True)
    parser.add_argument("--emi", type=float, default=None, help="Optional monthly EMI to assess.")
    parser.add_argument("--years", type=int, default=5, help="Loan tenure in years.")
    parser.add_argument(
        "--min-savings-rate",
        type=float,
        default=0.20,
        help="Required post-EMI monthly buffer as a fraction of income.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        predicted_expense = predict_expense(
            income=args.income,
            age=args.age,
            dependents=args.dependents,
            occupation=args.occupation,
            city_tier=args.city_tier,
        )
    except ValueError as error:
        raise SystemExit(f"Input error: {error}") from error

    print(f"Predicted Monthly Expense: {predicted_expense:.2f}")
    if args.emi is not None:
        affordability = assess_affordability(
            income=args.income,
            predicted_expense=predicted_expense,
            monthly_emi=args.emi,
            tenure_years=args.years,
            min_savings_rate=args.min_savings_rate,
        )
        print(f"Affordability Verdict: {affordability['verdict']}")
        print(f"Reason: {affordability['reason']}")
        print(f"Surplus After EMI: {affordability['surplus_after_emi']:.2f}")
        print(f"Required Monthly Buffer: {affordability['required_monthly_buffer']:.2f}")
        print(f"Total EMI Commitment: {affordability['total_emi_commitment']:.2f}")
