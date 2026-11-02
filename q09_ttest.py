import pandas as pd
from scipy import stats

df = pd.read_csv("Iris.csv")

setosa = df[df["Species"] == "Iris-setosa"]["SepalLengthCm"]
versicolor = df[df["Species"] == "Iris-versicolor"]["SepalLengthCm"]

print("Null Hypothesis (H0): Mean SepalLengthCm of setosa = Mean SepalLengthCm of versicolor")
print("Alternative Hypothesis (H1): Mean SepalLengthCm of setosa != Mean SepalLengthCm of versicolor")
print(f"Significance level: 0.05\n")

t_stat, p_value = stats.ttest_ind(setosa, versicolor)

print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.6f}")

if p_value < 0.05:
    print("\nConclusion: Reject the null hypothesis.")
    print("There is a significant difference in mean SepalLengthCm between setosa and versicolor.")
else:
    print("\nConclusion: Fail to reject the null hypothesis.")
    print("No significant difference in mean SepalLengthCm between setosa and versicolor.")
