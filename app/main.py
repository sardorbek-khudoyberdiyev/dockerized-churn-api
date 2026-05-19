from fastapi import FastAPI
from pydantic import BaseModel

from app.prediction import predict_churn

app = FastAPI(
    title="Customer Churn Prediction API",
    description="A machine learning API to predict customer churn based on input features.",
    version="1.0.0"
    )

class CustomerData(BaseModel):
    """Pydantic model for customer data input."""
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    SeniorCitizen: int = 0

    Contract_Month_to_month: int = 0
    InternetService_Fiber_optic: int = 0
    PaymentMethod_Electronic_check: int = 0
    PaperlessBilling_Yes: int = 0
    TechSupport_Yes: int = 0
    OnlineSecurity_Yes: int = 0


@app.get("/")
def home():
    """Root endpoint to check if the API is running."""
    return {"message": "Customer Churn Prediction API is running."}


@app.post("/predict")
def predict(customer: CustomerData):
    input_data = customer.model_dump()

    # Rename API-friendly field names to match training feature names
    input_data["Contract_Month-to-month"] = input_data.pop("Contract_Month_to_month")
    input_data["InternetService_Fiber optic"] = input_data.pop("InternetService_Fiber_optic")
    input_data["PaymentMethod_Electronic check"] = input_data.pop("PaymentMethod_Electronic_check")

    result = predict_churn(input_data)

    return result
