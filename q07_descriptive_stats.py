import pandas as pd

df = pd.read_csv("Iris.csv")

cols = ["SepalLengthCm", "PetalLengthCm"]

stats = pd.DataFrame({
    "Mean": df[cols].mean(),
    "Median": df[cols].median(),
    "Variance": df[cols].var(),
    "Std Dev": df[cols].std()
})

print("Descriptive Statistics:")
print(stats)

print("\nObservation:")
print("PetalLengthCm has a larger variance and standard deviation than SepalLengthCm,")
print("indicating that petal lengths are more spread out across the dataset.")
