import pandas as pd

df = pd.read_csv("Iris.csv")

print("First five records:")
print(df.head())

print("\nNumber of rows and columns:", df.shape)
print("\nColumn names:", list(df.columns))
print("\nData types:")
print(df.dtypes)

print("\nSummary statistics:")
print(df.describe())

print("\nNumerical attributes: SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm")
print("Target column: Species")
