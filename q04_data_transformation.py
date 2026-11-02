import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Iris.csv")

# One-hot encoding of Species
df_encoded = pd.get_dummies(df, columns=["Species"], dtype=int)
print("After one-hot encoding:")
print(df_encoded.head())

# Standardize numerical attributes
scaler = StandardScaler()
num_cols = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
df_encoded[num_cols] = scaler.fit_transform(df_encoded[num_cols])

print("\nAfter standardization:")
print(df_encoded.head())
