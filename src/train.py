
import os
import json
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load training and testing data
train_data = pd.read_csv("data/processed/train.csv")
test_data = pd.read_csv("data/processed/test.csv")

# Separate features and target
X_train = train_data.drop(columns=["target"])
y_train = train_data["target"]

X_test = test_data.drop(columns=["target"])
y_test = test_data["target"]

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

# Set MLflow tracking
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Iris_Classification")

# Train and track model
with mlflow.start_run():

    # Train model
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)

    print("\nModel Accuracy:", accuracy)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=[
            "Setosa",
            "Versicolor",
            "Virginica"
        ]
    ))

    # Confusion matrix
    cm = confusion_matrix(y_test, predictions)

    print("\nConfusion Matrix:")
    print(cm)

    # Log parameters and metrics
    mlflow.log_param("algorithm", "Random Forest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)

    mlflow.log_metric("accuracy", accuracy)

    # Save trained model
    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/iris_model.pkl")

    # Save metrics
    with open("metrics.json", "w") as file:
        json.dump(
            {"accuracy": float(accuracy)},
            file,
            indent=4
        )

    # Log model in MLflow
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="iris_model",
        skops_trusted_types=[
            "sklearn.tree._tree.Tree"
        ]
    )

print("\nModel training completed successfully!")
print("Model saved in models/iris_model.pkl")
