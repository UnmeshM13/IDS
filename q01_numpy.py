import numpy as np
import pandas as pd

df = pd.read_csv("Iris.csv")

sepal_length = np.array(df["SepalLengthCm"])

print("First five values:", sepal_length[:5])
print("Sum:", np.sum(sepal_length))
print("Mean:", np.mean(sepal_length))
print("Maximum:", np.max(sepal_length))
print("Minimum:", np.min(sepal_length))
