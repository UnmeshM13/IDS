import pandas as pd

df = pd.read_csv("Iris.csv")

corr = df["PetalLengthCm"].corr(df["PetalWidthCm"])
cov = df["PetalLengthCm"].cov(df["PetalWidthCm"])

print(f"Correlation coefficient: {corr:.4f}")
print(f"Covariance: {cov:.4f}")

print("\nObservation:")
print("The correlation is strongly positive (close to 1), indicating that as")
print("PetalLengthCm increases, PetalWidthCm also increases proportionally.")
print("The positive covariance confirms the same direction of variation.")
