import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

# ---------------- LOAD BUILT-IN DATASET ----------------
wine = load_wine()

# Use only first 5 features (easy to explain)
X = wine.data[:, :5]

# ---------------- FEATURE SCALING ----------------
scaler = StandardScaler()
X = scaler.fit_transform(X)

print("Dataset Shape:", X.shape)

# ---------------- EUCLIDEAN DISTANCE ----------------
def euclidean(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# ---------------- HIERARCHICAL CLUSTERING ----------------
clusters = [[i] for i in range(len(X))]

step = 1

while len(clusters) > 3:   # Stop when 3 clusters remain
    min_dist = float('inf')
    pair = (0, 1)

    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            for p in clusters[i]:
                for q in clusters[j]:
                    dist = euclidean(X[p], X[q])
                    if dist < min_dist:
                        min_dist = dist
                        pair = (i, j)

    # Merge closest clusters
    clusters[pair[0]] += clusters[pair[1]]
    clusters.pop(pair[1])

    print(f"\nStep {step} - Merged clusters {pair}")
    print("Number of clusters:", len(clusters))
    step += 1

print("\nFinal Clusters:")
for i, cluster in enumerate(clusters):
    print(f"Cluster {i+1}:", cluster)