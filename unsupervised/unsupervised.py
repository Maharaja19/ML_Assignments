import streamlit as st
import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import random

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(page_title="Unsupervised Learning Dashboard", layout="wide")

# ---------------- TITLE ----------------
st.title("📊 Unsupervised Learning Dashboard")
st.write("Grouping wine samples using Hierarchical and K-Means Clustering")

# ---------------- LOAD DATA ----------------
wine = load_wine()

# Use more meaningful features
X = wine.data[:, :4]   # alcohol, malic acid, ash, alcalinity

# Scale data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ---------------- DISTANCE FUNCTION ----------------
def euclidean(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# =====================================================
# 🟢 HIERARCHICAL CLUSTERING
# =====================================================
def hierarchical_clustering(X, k=3):
    clusters = [[i] for i in range(len(X))]

    while len(clusters) > k:
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

        clusters[pair[0]] += clusters[pair[1]]
        clusters.pop(pair[1])

    return clusters

# =====================================================
# 🟡 K-MEANS CLUSTERING
# =====================================================
def kmeans(X, k=3):
    centroids = X[random.sample(range(len(X)), k)]

    for _ in range(100):
        clusters = [[] for _ in range(k)]

        for idx, point in enumerate(X):
            distances = [euclidean(point, c) for c in centroids]
            cluster_index = np.argmin(distances)
            clusters[cluster_index].append(idx)

        new_centroids = []
        for cluster in clusters:
            new_centroids.append(np.mean(X[cluster], axis=0))

        new_centroids = np.array(new_centroids)

        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return clusters, centroids

# =====================================================
# 🎛️ INPUT
# =====================================================
k = st.slider("Select Number of Clusters (k)", 2, 5, 3)

st.subheader("📄 Sample Data")
st.dataframe(X[:10])

# =====================================================
# 🚀 RUN
# =====================================================
if st.button("Run Clustering"):

    h_clusters = hierarchical_clustering(X, k)
    k_clusters, centroids = kmeans(X, k)

    # ---------------- OUTPUT ----------------
    st.subheader("🔍 Clustering Results")

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Hierarchical Clustering")
        for i, cluster in enumerate(h_clusters):
            st.write(f"Cluster {i+1}: {len(cluster)} samples")

    with col2:
        st.write("### K-Means Clustering")
        for i, cluster in enumerate(k_clusters):
            st.write(f"Cluster {i+1}: {len(cluster)} samples")

    # =====================================================
    # 📊 GRAPHS
    # =====================================================

    st.subheader("📊 Visualization")

    # 1️⃣ Hierarchical Scatter
    fig1, ax1 = plt.subplots()
    for cluster in h_clusters:
        points = X[cluster]
        ax1.scatter(points[:, 0], points[:, 1])
    ax1.set_title("Hierarchical Clustering")
    st.pyplot(fig1)

    # 2️⃣ K-Means Scatter
    fig2, ax2 = plt.subplots()
    for cluster in k_clusters:
        points = X[cluster]
        ax2.scatter(points[:, 0], points[:, 1])
    ax2.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200)
    ax2.set_title("K-Means Clustering")
    st.pyplot(fig2)

    # 3️⃣ Cluster Size Comparison
    fig3, ax3 = plt.subplots()
    ax3.bar(range(len(h_clusters)), [len(c) for c in h_clusters])
    ax3.set_title("Hierarchical Cluster Sizes")
    st.pyplot(fig3)

    # 4️⃣ K-Means Cluster Size
    fig4, ax4 = plt.subplots()
    ax4.bar(range(len(k_clusters)), [len(c) for c in k_clusters])
    ax4.set_title("K-Means Cluster Sizes")
    st.pyplot(fig4)

    # =====================================================
    # ✅ CONCLUSION
    # =====================================================
    st.subheader("✅ Conclusion")

    st.success("""
    - The system groups wine samples based on similarity.
    - K-Means forms clusters using centroids.
    - Hierarchical clustering builds clusters step-by-step.
    - No labeled data is used (unsupervised learning).
    """)