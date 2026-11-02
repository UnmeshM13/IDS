import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Iris.csv")

# Box plot for SepalWidthCm
plt.figure(figsize=(6, 5))
plt.boxplot(df["SepalWidthCm"], vert=True)
plt.title("Box Plot of SepalWidthCm")
plt.ylabel("SepalWidthCm")
plt.tight_layout()
# plt.savefig("q06_boxplot.png")
plt.show()

# Calculate Q1, Q3, IQR
Q1 = df["SepalWidthCm"].quantile(0.25)
Q3 = df["SepalWidthCm"].quantile(0.75)
IQR = Q3 - Q1

print(f"Q1: {Q1}")
print(f"Q3: {Q3}")
print(f"IQR: {IQR}")

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
print(f"Lower bound: {lower_bound}")
print(f"Upper bound: {upper_bound}")

outliers = df[(df["SepalWidthCm"] < lower_bound) | (df["SepalWidthCm"] > upper_bound)]
print(f"\nNumber of outliers: {len(outliers)}")
print("Outlier values:")
print(outliers[["Id", "SepalWidthCm"]])
