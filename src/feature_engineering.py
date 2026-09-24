import pandas as pd

# Load dataset
df = pd.read_csv("dataset/crop_yield_dataset.csv")

print("===================================")
print("     FEATURE ENGINEERING")
print("===================================")

# Display original data
print("\nOriginal Dataset:")
print(df.head())


# Convert categorical columns into numerical values
df_encoded = pd.get_dummies(
    df,
    columns=["Crop", "Soil_Type"],
    drop_first=True
)


print("\nEncoded Dataset:")
print(df_encoded.head())


# Separate features and target
X = df_encoded.drop("Yield", axis=1)
y = df_encoded["Yield"]


print("\nFeatures (X):")
print(X.columns.tolist())

print("\nTarget (y):")
print("Yield")


print("\nX Shape:", X.shape)
print("y Shape:", y.shape)


# Save processed dataset
df_encoded.to_csv(
    "dataset/processed_crop_yield.csv",
    index=False
)

print("\nProcessed dataset saved successfully.")
print("File: dataset/processed_crop_yield.csv")