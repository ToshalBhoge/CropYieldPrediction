import pandas as pd
from pathlib import Path

# -----------------------------
# 1. Load Dataset
# -----------------------------

project_folder = Path(__file__).resolve().parent.parent
dataset_path = project_folder / "dataset" / "crop_yield_dataset.csv"

df = pd.read_csv(dataset_path)

print("===================================")
print("     CROP YIELD DATASET ANALYSIS")
print("===================================")

# -----------------------------
# 2. First 5 Rows
# -----------------------------

print("\n1. First 5 Rows:")
print(df.head())

# -----------------------------
# 3. Last 5 Rows
# -----------------------------

print("\n2. Last 5 Rows:")
print(df.tail())

# -----------------------------
# 4. Dataset Shape
# -----------------------------

print("\n3. Dataset Shape:")
print(df.shape)

# -----------------------------
# 5. Column Names
# -----------------------------

print("\n4. Column Names:")
print(df.columns.tolist())

# -----------------------------
# 6. Data Types
# -----------------------------

print("\n5. Data Types:")
print(df.dtypes)

# -----------------------------
# 7. Missing Values
# -----------------------------

print("\n6. Missing Values:")
print(df.isnull().sum())

# -----------------------------
# 8. Duplicate Rows
# -----------------------------

print("\n7. Duplicate Rows:")
print(df.duplicated().sum())

# -----------------------------
# 9. Statistical Summary
# -----------------------------

print("\n8. Statistical Summary:")
print(df.describe())