import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv("Iris.csv")

X = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X)

print("Cluster assignments:")
print(df[["Id", "Species", "Cluster"]].to_string(index=False))

# Scatter plot using two attributes
plt.figure(figsize=(8, 5))
colors = {0: "red", 1: "green", 2: "blue"}
for cluster in [0, 1, 2]:
    group = df[df["Cluster"] == cluster]
    plt.scatter(group["PetalLengthCm"], group["PetalWidthCm"],
                label=f"Cluster {cluster}", color=colors[cluster], alpha=0.7)
plt.title("K-Means Clustering (PetalLengthCm vs PetalWidthCm)")
plt.xlabel("PetalLengthCm")
plt.ylabel("PetalWidthCm")
# plt.legend()
# plt.tight_layout()
# plt.savefig("q12_kmeans.png")
plt.show()
