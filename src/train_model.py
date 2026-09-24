import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import numpy as np

print("===================================")
print("     CROP YIELD MODEL TRAINING")
print("===================================")

# Load processed dataset
df = pd.read_csv("dataset/processed_crop_yield.csv")

print("\nProcessed dataset loaded!")
print("Shape:", df.shape)

# Separate features and target
X = df.drop("Yield", axis=1)
y = df["Yield"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

# Create Random Forest model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

print("\nModel training completed!")

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n===================================")
print("       MODEL EVALUATION")
print("===================================")

print(f"\nMean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score: {r2:.4f}")
print(f"R² Score (%): {r2 * 100:.2f}%")

# Save model
joblib.dump(model, "model.pkl")

# Save feature names
joblib.dump(list(X.columns), "feature_columns.pkl")

print("\nModel saved successfully!")
print("File: model.pkl")

print("\nFeature columns saved successfully!")
print("File: feature_columns.pkl")