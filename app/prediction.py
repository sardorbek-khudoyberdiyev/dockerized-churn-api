import joblib
import pandas as pd

def load_artifacts():
    """Load the trained model and feature names from disk."""
    model = joblib.load("models/churn_model.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    return model, feature_names

model, feature_names = load_artifacts()

def predict_churn(input_data):
    """Predict churn based on input data."""
    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=feature_names, fill_value=0)
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    results = {
        "prediction" : "Churn" if prediction == 1 else "No Churn",
        "churn_probability": round(float(probability), 4)
    }

    return results