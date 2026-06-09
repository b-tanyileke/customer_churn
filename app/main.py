"""FastAPI app for telco customer churn predictions."""

import sys
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from pydantic import Field

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from predict import predict_customer


app = FastAPI(title="Telco Churn Prediction API")


class CustomerRequest(BaseModel):
    """Raw customer fields used by the training pipeline."""

    dependents: str = Field(alias="Dependents")
    married: str = Field(alias="Married")
    senior_citizen: str = Field(alias="Senior Citizen")
    satisfaction_score: int = Field(alias="Satisfaction Score")
    cltv: int = Field(alias="CLTV")
    internet_service: str = Field(alias="Internet Service")
    online_security: str = Field(alias="Online Security")
    premium_tech_support: str = Field(alias="Premium Tech Support")
    contract: str = Field(alias="Contract")
    paperless_billing: str = Field(alias="Paperless Billing")
    monthly_charge: float = Field(alias="Monthly Charge")
    tenure_in_months: int = Field(alias="Tenure in Months")


def request_to_dataframe(customer: CustomerRequest) -> pd.DataFrame:
    """Converts one API request into a DataFrame for the ML pipeline."""
    if hasattr(customer, "model_dump"):
        data = customer.model_dump(by_alias=True)
    else:
        data = customer.dict(by_alias=True)

    return pd.DataFrame([data])


@app.get("/")
def home() -> dict[str, str]:
    """Returns a simple API welcome message."""
    return {"message": "Telco Churn Prediction API"}


@app.get("/health")
def health() -> dict[str, str]:
    """Returns the API health status."""
    return {"status": "healthy"}


@app.post("/predict")
def predict(customer: CustomerRequest) -> dict[str, str | float | list[dict[str, str | float]]]:
    """Predicts churn for a single customer record."""
    features = request_to_dataframe(customer)
    result = predict_customer(features)

    return {
        "model_name": result["model_name"],
        "model_version": result["model_version"],
        "prediction": result["prediction"],
        "probability": result["probability"],
        "risk_level": result["risk_level"],
        "top_factors": result["top_factors"]
    }
