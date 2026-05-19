import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, recall_score


MODEL_PATH = "models/churn_model.pkl"
DATA_PATH = "data/processed/model_ready_data.csv"
FEATURE_NAMES_PATH = "models/feature_names.pkl"


def load_data():
    """Load the dataset from a CSV file."""
    return pd.read_csv(DATA_PATH)

def split_features_target(df):
    """Split the DataFrame into features and target variable."""
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    return X, y

def train_model(X_train, y_train):
    """Train a Random Forest Classifier."""
    model = RandomForestClassifier(n_estimators=200, 
                                   random_state=42,
                                   max_depth=None,
                                   min_samples_split=5,
                                   min_samples_leaf=1,
                                   class_weight='balanced')
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print performance metrics."""
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    
    print("Model Evaluation:")
    print("------------------")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

def save_model(model, feature_names):
    """Save the trained model and feature names to disk."""
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(feature_names, FEATURE_NAMES_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Feature names saved to {FEATURE_NAMES_PATH}")

def main():
    # Load and prepare data
    print("Loading data...")
    df = load_data()
    print("Splitting features and target variable...")
    X, y = split_features_target(df)

    # Split into training and testing sets
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    print("Training the model...")
    model = train_model(X_train, y_train)

    # Evaluate the model
    print("Evaluating the model...")
    evaluate_model(model, X_test, y_test)

    # Save the model and feature names
    print("Saving the model and feature names...")
    save_model(model, X.columns.tolist())

    print("Training process completed successfully.")

if __name__ == "__main__":
    main()