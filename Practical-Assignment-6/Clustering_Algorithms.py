import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score
)

# --------------------------------------------------
# 1. Load the Wine Dataset
# --------------------------------------------------
data = load_wine()
X = data.data
y_true = data.target  # Used only for external validation (ARI)

# --------------------------------------------------
# 2. Feature Scaling
# --------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------------------------------
# 3. Select Number of Clusters for K-Means
# --------------------------------------------------
print("K-MEANS: SELECTION OF NUMBER OF CLUSTERS")
print("-" * 60)

for k in range(2, 9):
    km = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42
    ).fit(X_scaled)

    print(
        f"k={k}  "
        f"Inertia={km.inertia_:.2f}  "
        f"Silhouette={silhouette_score(X_scaled, km.labels_):.4f}  "
        f"DBI={davies_bouldin_score(X_scaled, km.labels_):.4f}"
    )

# --------------------------------------------------
# 4. Create Clustering Models
# --------------------------------------------------
models = {
    "K-Means": KMeans(
        n_clusters=3,
        n_init=10,
        random_state=42
    ),
    "Hierarchical": AgglomerativeClustering(
        n_clusters=3,
        linkage="ward"
    ),
    "GMM": GaussianMixture(
        n_components=3,
        random_state=42
    ),
    "DBSCAN": DBSCAN(
        eps=2.3,
        min_samples=5
    )
}

# --------------------------------------------------
# 5. Fit Models and Evaluate Performance
# --------------------------------------------------
results = []
all_labels = {}

for name, model in models.items():

    labels = model.fit_predict(X_scaled)
    all_labels[name] = labels

    # Ignore noise points (-1) for internal evaluation
    mask = labels != -1
    valid_labels = labels[mask]
    X_valid = X_scaled[mask]

    n_clusters = len(set(valid_labels))
    n_noise = int((~mask).sum())

    # Calculate metrics only when at least two clusters exist
    # and each cluster is not a separate individual sample.
    if 1 < n_clusters < len(X_valid):
        silhouette = round(
            silhouette_score(X_valid, valid_labels), 4
        )
        dbi = round(
            davies_bouldin_score(X_valid, valid_labels), 4
        )
        ch_score = round(
            calinski_harabasz_score(X_valid, valid_labels), 2
        )
    else:
        silhouette = np.nan
        dbi = np.nan
        ch_score = np.nan

    ari = round(adjusted_rand_score(y_true, labels), 4)

    results.append({
        "Algorithm": name,
        "Clusters": n_clusters,
        "Noise": n_noise,
        "Silhouette": silhouette,
        "Davies-Bouldin": dbi,
        "Calinski-Harabasz": ch_score,
        "ARI": ari
    })

# --------------------------------------------------
# 6. Display Comparison Table
# --------------------------------------------------
df = pd.DataFrame(results)

print("\nCLUSTERING ALGORITHM COMPARISON")
print("-" * 85)
print(df.to_string(index=False))

# --------------------------------------------------
# 7. Display K-Means Cluster Sizes
# --------------------------------------------------
print("\nK-MEANS CLUSTER SIZES:")
print(
    dict(
        sorted(
            pd.Series(all_labels["K-Means"])
            .value_counts()
            .items()
        )
    )
)

# --------------------------------------------------
# 8. Performance Summary
# --------------------------------------------------
print("\nPERFORMANCE SUMMARY")

valid_silhouette = df.dropna(subset=["Silhouette"])
valid_dbi = df.dropna(subset=["Davies-Bouldin"])

if not valid_silhouette.empty:
    print(
        "Highest Silhouette Score:",
        valid_silhouette.loc[
            valid_silhouette["Silhouette"].idxmax(),
            "Algorithm"
        ]
    )

if not valid_dbi.empty:
    print(
        "Lowest Davies-Bouldin Index:",
        valid_dbi.loc[
            valid_dbi["Davies-Bouldin"].idxmin(),
            "Algorithm"
        ]
    )

print(
    "Highest ARI:",
    df.loc[df["ARI"].idxmax(), "Algorithm"]
)

# --------------------------------------------------
# 9. Visualize Clusters Using PCA
# --------------------------------------------------
X_2d = PCA(
    n_components=2,
    random_state=42
).fit_transform(X_scaled)

fig, axes = plt.subplots(1, 4, figsize=(18, 4))

for ax, (name, labels) in zip(axes, all_labels.items()):
    ax.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=labels,
        cmap="viridis",
        s=20
    )

    ax.set_title(name)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

plt.tight_layout()

# Save the visualization
plt.savefig("clustering_comparison.png", dpi=150)

# Display the plots
plt.show()

print("\nProgram executed successfully!")
