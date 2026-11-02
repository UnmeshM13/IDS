import pandas as pd
import numpy as np

df = pd.read_csv("Iris.csv")
df_copy = df.copy()

# Introduce missing values
df_copy.loc[2, "SepalLengthCm"] = np.nan
df_copy.loc[5, "SepalLengthCm"] = np.nan
df_copy.loc[10, "SepalLengthCm"] = np.nan
df_copy.loc[7, "Species"] = np.nan
df_copy.loc[12, "Species"] = np.nan

print("Missing values before handling:")
print(df_copy.isnull().sum())

# Fill numerical column with median
median_val = df_copy["SepalLengthCm"].median()
df_copy["SepalLengthCm"] = df_copy["SepalLengthCm"].fillna(median_val)
print("\nSepalLengthCm filled with median:", median_val)

# Fill categorical column with mode
mode_val = df_copy["Species"].mode()[0]
df_copy["Species"] = df_copy["Species"].fillna(mode_val)
print("Species filled with mode:", mode_val)

print("\nMissing values after handling:")
print(df_copy.isnull().sum())
