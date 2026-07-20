"""FastAPI backend for expense and affordability predictions."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from predict import assess_affordability, predict_expense


class PredictionRequest(BaseModel):
    """Request payload for affordability prediction."""

    income: float = Field(..., gt=0)
    age: int = Field(..., ge=18, le=100)
    dependents: int = Field(..., ge=0, le=10)
    occupation: str = Field(..., min_length=1)
    city_tier: str = Field(..., min_length=1)
    monthly_emi: float = Field(0, ge=0)
    tenure_years: int = Field(5, ge=1, le=40)
    min_savings_rate: float = Field(0.20, ge=0, lt=1)


class PredictionResponse(BaseModel):
    """Response payload returned to the frontend."""

    predicted_monthly_expense: float
    affordability: dict[str, Any]


app = FastAPI(title="Expense Affordability API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    """Predict monthly expense and assess EMI affordability."""
    try:
        expense = predict_expense(
            income=payload.income,
            age=payload.age,
            dependents=payload.dependents,
            occupation=payload.occupation,
            city_tier=payload.city_tier,
        )
        affordability = assess_affordability(
            income=payload.income,
            predicted_expense=expense,
            monthly_emi=payload.monthly_emi,
            tenure_years=payload.tenure_years,
            min_savings_rate=payload.min_savings_rate,
        )
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return PredictionResponse(
        predicted_monthly_expense=expense,
        affordability=affordability,
    )
