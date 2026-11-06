
import pandas as pd
from sklearn.datasets import load_iris
import os

# Load Iris dataset
iris = load_iris()

# Convert dataset into DataFrame
df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# Add target column
df["target"] = iris.target

# Add species names
df["species"] = [
    iris.target_names[i] for i in iris.target
]

# Create raw data folder
os.makedirs("data/raw", exist_ok=True)

# Save dataset
df.to_csv("data/raw/iris.csv", index=False)

print("Iris dataset created successfully!")

print("\nFirst five records:")
print(df.head())

print("\nDataset shape:", df.shape)

print("\nSpecies distribution:")
print(df["species"].value_counts())

