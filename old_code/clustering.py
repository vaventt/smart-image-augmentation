import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


# Constants for directory paths
BASE_DIR = "/Users/andrew/Thesis/smart-image-augmentation"
FILTRATION_CSV_PATH = os.path.join(BASE_DIR, "results", "filtration-results", "pascal-0-1-results.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "clustering_analysis")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the data
df = pd.read_csv(FILTRATION_CSV_PATH)

# Select the columns for clustering
SCORE_COLUMNS = ['class_representation', 'visual_fidelity', 'structural_integrity', 'diversity', 'beneficial_variations']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[SCORE_COLUMNS])

# Function to compute WCSS (Within-Cluster Sum of Square)
def compute_wcss(data):
    wcss = []
    for n in range(4, 21):
        kmeans = KMeans(n_clusters=n, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    return wcss

# Compute WCSS for elbow method
wcss = compute_wcss(X_scaled)

# Compute silhouette scores
silhouette_scores = []
for n in range(4, 21):
    kmeans = KMeans(n_clusters=n, init='k-means++', max_iter=300, n_init=10, random_state=0)
    cluster_labels = kmeans.fit_predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    print(f"For n_clusters = {n}, the average silhouette score is : {silhouette_avg}")

# Find optimal eps
neighbors = NearestNeighbors(n_neighbors=10)
nbrs = neighbors.fit(X_scaled)
distances, indices = nbrs.kneighbors(X_scaled)
distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.plot(distances)
plt.title('K-distance Graph')
plt.xlabel('Data Points sorted by distance')
plt.ylabel('Epsilon')
plt.savefig(os.path.join(OUTPUT_DIR, 'dbscan_epsilon.png'))
plt.close()

# Choose eps from the elbow of this plot

# Then, for different min_samples:
for min_samples in range(2, 10):
    dbscan = DBSCAN(eps=0.4, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(X_scaled)
    if len(set(cluster_labels)) > 1:  # Silhouette Score requires at least 2 clusters
        score = silhouette_score(X_scaled, cluster_labels)
        print(f"DBSCAN with min_samples={min_samples}, Silhouette Score: {score}")

# Plot elbow curve
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(4, 21), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')

# Plot silhouette scores
plt.subplot(1, 2, 2)
plt.plot(range(4, 21), silhouette_scores)
plt.title('Silhouette Score Method')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'clustering_analysis.png'))
plt.close()

print(f"Clustering analysis plot saved to {os.path.join(OUTPUT_DIR, 'clustering_analysis.png')}")

# Find the optimal number of clusters
optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 4
print(f"The optimal number of clusters based on silhouette score is: {optimal_clusters}")