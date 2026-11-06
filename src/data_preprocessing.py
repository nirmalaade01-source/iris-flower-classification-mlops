
import pandas as pd
import os

from sklearn.model_selection import train_test_split

def preprocess_data():

    # Load raw dataset
    df = pd.read_csv("data/raw/iris.csv")

    print("Original Dataset Shape:", df.shape)

    # Separate input features
    X = df.drop(columns=["target", "species"])

    # Separate target variable
    y = df["target"]

    # Split dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Create processed folder
    os.makedirs("data/processed", exist_ok=True)

    # Save training dataset
    train_data = X_train.copy()
    train_data["target"] = y_train

    train_data.to_csv(
        "data/processed/train.csv",
        index=False
    )

    # Save testing dataset
    test_data = X_test.copy()
    test_data["target"] = y_test

    test_data.to_csv(
        "data/processed/test.csv",
        index=False
    )

    print("\nData preprocessing completed!")

    print("Training data shape:", X_train.shape)
    print("Testing data shape:", X_test.shape)

    print("\nTraining dataset saved successfully!")
    print("Testing dataset saved successfully!")

if __name__ == "__main__":
    preprocess_data()
