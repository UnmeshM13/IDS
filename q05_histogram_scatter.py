import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Iris.csv")

# Histogram of PetalLengthCm
plt.figure(figsize=(8, 5))
plt.hist(df["PetalLengthCm"], bins=20, color="steelblue", edgecolor="black")
plt.title("Histogram of PetalLengthCm")
plt.xlabel("PetalLengthCm")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("q05_histogram.png")
plt.show()

# Scatter plot of SepalLengthCm vs PetalLengthCm coloured by Species
plt.figure(figsize=(8, 5))
colors = {"Iris-setosa": "red", "Iris-versicolor": "green", "Iris-virginica": "blue"}
for species, group in df.groupby("Species"):
    plt.scatter(group["SepalLengthCm"], group["PetalLengthCm"],
                label=species, color=colors[species], alpha=0.7)
plt.title("SepalLengthCm vs PetalLengthCm")
plt.xlabel("SepalLengthCm")
plt.ylabel("PetalLengthCm")
plt.legend()
plt.tight_layout()
plt.savefig("q05_scatter.png")
plt.show()

print("Observation:")
print("The histogram shows a bimodal distribution — setosa has shorter petals while")
print("versicolor and virginica have longer petals.")
print("The scatter plot shows a positive relationship between SepalLengthCm and PetalLengthCm,")
print("with clear separation among the three species.")
