# Step 1: Load the Dataset

import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.cluster.hierarchy import linkage, dendrogram

url = "https://raw.githubusercontent.com/miltonsuggs/11-mall-customer-segmentation/main/Mall_Customers.csv"
df = pd.read_csv(url).rename(columns={
   "Annual Income (k$)": "Income",
   "Spending Score (1-100)": "Score"
})
print(df.head())
print(df[["Age", "Income", "Score"]].describe().round(1))


# Step 2: Select and Scale the Features

features = ["Income", "Score"]
X = StandardScaler().fit_transform(df[features])
print("Scaled data shape:", X.shape)


# Step 3: Find k using the Elbow Method and Silhouette Score

inertia, sil = [], []
ks = range(2, 11)
for k in ks:
   km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
   inertia.append(km.inertia_)
   sil.append(silhouette_score(X, km.labels_))

print(pd.DataFrame({
   "k": list(ks),
   "inertia": np.round(inertia, 1),
   "silhouette": np.round(sil, 3)
}))
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].plot(ks, inertia, marker="o")
ax[0].set_title("Elbow method")
ax[0].set_xlabel("k")
ax[0].set_ylabel("Inertia")
ax[1].plot(ks, sil, marker="o", color="green")
ax[1].set_title("Silhouette score")
ax[1].set_xlabel("k")
plt.tight_layout()
plt.show()


# Step 4: Build the K-Means Model (k = 5)

kmeans = KMeans(n_clusters=5, n_init=10, random_state=42)
df["Cluster"] = kmeans.fit_predict(X)
print(df["Cluster"].value_counts().sort_index())
print("Silhouette score:", round(silhouette_score(X, df["Cluster"]), 3))


# Step 5: Visualize the Clusters

centers = StandardScaler().fit(df[features]).inverse_transform(
   kmeans.cluster_centers_
)
plt.figure(figsize=(7, 5))
sns.scatterplot(
   data=df, x="Income", y="Score",
   hue="Cluster", palette="Set1", s=60
)

plt.scatter(
   centers[:, 0], centers[:, 1],
   c="black", marker="X", s=200, label="Centroids"
)
plt.title("Customer segments (K-Means, k = 5)")
plt.legend()
plt.show()


# Step 6: Profile the Segments

profile = df.groupby("Cluster")[["Age", "Income", "Score"]].mean().round(1)
profile["Customers"] = df["Cluster"].value_counts().sort_index()
print(profile)


# Step 7: Hierarchical Clustering

Z = linkage(X, method="ward")
plt.figure(figsize=(10, 4))
dendrogram(Z, truncate_mode="lastp", p=20, leaf_rotation=90)
plt.title("Hierarchical clustering dendrogram (Ward)")
plt.xlabel("Clusters")
plt.ylabel("Distance")
plt.show()

hier = AgglomerativeClustering(
   n_clusters=5, linkage="ward"
).fit_predict(X)
print(
   "Agreement with K-Means (adjusted Rand index):",
   round(adjusted_rand_score(df["Cluster"], hier), 3)
)
