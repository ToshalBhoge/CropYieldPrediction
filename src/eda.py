import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("dataset/crop_yield_dataset(1).csv")

print("===================================")
print("        CROP YIELD EDA")
print("===================================")

print("\nDataset loaded successfully!")
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# 1. Crop Distribution
plt.figure(figsize=(8, 5))

sns.countplot(data=df, x="Crop")

plt.title("Crop Distribution")
plt.xlabel("Crop")
plt.ylabel("Number of Records")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# 2. Yield Distribution
plt.figure(figsize=(8, 5))

sns.histplot(data=df, x="Yield", bins=20, kde=True)

plt.title("Yield Distribution")
plt.xlabel("Yield")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


# 3. Rainfall vs Yield
plt.figure(figsize=(8, 5))

sns.scatterplot(data=df, x="Rainfall", y="Yield")

plt.title("Rainfall vs Crop Yield")
plt.xlabel("Rainfall")
plt.ylabel("Yield")

plt.tight_layout()
plt.show()


# 4. Temperature vs Yield
plt.figure(figsize=(8, 5))

sns.scatterplot(data=df, x="Temperature", y="Yield")

plt.title("Temperature vs Crop Yield")
plt.xlabel("Temperature")
plt.ylabel("Yield")

plt.tight_layout()
plt.show()


# 5. Fertilizer vs Yield
plt.figure(figsize=(8, 5))

sns.scatterplot(data=df, x="Fertilizer", y="Yield")

plt.title("Fertilizer vs Crop Yield")
plt.xlabel("Fertilizer")
plt.ylabel("Yield")

plt.tight_layout()
plt.show()


# 6. Pesticide vs Yield
plt.figure(figsize=(8, 5))

sns.scatterplot(data=df, x="Pesticide", y="Yield")

plt.title("Pesticide vs Crop Yield")
plt.xlabel("Pesticide")
plt.ylabel("Yield")

plt.tight_layout()
plt.show()


# 7. Correlation Heatmap
numeric_df = df.select_dtypes(include=["int64", "float64"])

plt.figure(figsize=(10, 7))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.show()


print("\nEDA completed successfully.")