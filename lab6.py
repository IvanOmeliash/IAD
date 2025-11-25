import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from matplotlib.patches import Circle

# --- 1. ПІДГОТОВКА ДАНИХ ---
data = {
    'X1': [5, 4, 6, 80, 95, 70, 40, 50, 10, 15, 90, 20, 12, 100],
    'X2': [150, 140, 160, 350, 400, 320, 250, 280, 200, 180, 380, 200, 170, 450]
}
df = pd.DataFrame(data)
X = df[['X1', 'X2']].values
N = X.shape[0]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 2. ФУНКЦІЯ K-MEANS ---
def kmeans_final(X_scaled, K, max_iter=100):
    np.random.seed(42)
    initial_indices = np.random.choice(len(X_scaled), K, replace=False)
    centroids = X_scaled[initial_indices]

    for _ in range(max_iter):
        distances = cdist(X_scaled, centroids, 'euclidean')
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([X_scaled[labels == k].mean(axis=0) if np.any(labels == k) else centroids[k]
                                  for k in range(K)])
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return centroids, labels

# --- 3. ФУНКЦІЯ DUNN INDEX ---
def dunn_index(X, labels):
    clusters = np.unique(labels)
    # Мінімальна міжкластерна відстань
    min_intercluster = np.inf
    for i in range(len(clusters)):
        for j in range(i+1, len(clusters)):
            points_i = X[labels == clusters[i]]
            points_j = X[labels == clusters[j]]
            dist_ij = cdist(points_i, points_j, 'euclidean')
            min_dist = dist_ij.min()
            if min_dist < min_intercluster:
                min_intercluster = min_dist
    # Максимальний діаметр кластера
    max_intracluster = 0
    for k in clusters:
        points_k = X[labels == k]
        if len(points_k) > 1:
            dist_k = cdist(points_k, points_k, 'euclidean')
            diam = dist_k.max()
            if diam > max_intracluster:
                max_intracluster = diam
    return min_intercluster / max_intracluster if max_intracluster > 0 else 0

# --- 4. ВІЗУАЛІЗАЦІЯ ФІНАЛЬНИХ КЛАСТЕРІВ ---
def plot_clusters(X, labels, centroids_orig, K, radii, dunn):
    plt.figure(figsize=(10, 7))
    colors = plt.cm.viridis(np.linspace(0, 1, K))
    for k in range(K):
        points = X[labels == k]
        plt.scatter(points[:, 0], points[:, 1], s=100, color=colors[k], alpha=0.7, edgecolor='k')
        plt.scatter(centroids_orig[k, 0], centroids_orig[k, 1], marker='X', s=250,
                    color='red', edgecolor='k', linewidths=2)
        if radii[k] > 0:
            circle = Circle(centroids_orig[k], radii[k], color=colors[k], fill=False,
                            linestyle='--', linewidth=1.5, alpha=0.6)
            plt.gca().add_patch(circle)
    plt.title(f'K-Means K={K} | Dunn Index = {dunn:.4f}')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

# --- 5. ЦИКЛ ПО K=2..N-1 + ЗБІР DUNN INDEX ---
dunn_values = []
K_values = list(range(2, int(N/2)))

for K in K_values:
    final_centroids_scaled, final_labels = kmeans_final(X_scaled, K)
    final_centroids_orig = scaler.inverse_transform(final_centroids_scaled)

    # Радіуси кластерів
    radii = []
    for k in range(K):
        points_in_cluster = X[final_labels == k]
        if len(points_in_cluster) > 0:
            cluster_centroid_orig = final_centroids_orig[k].reshape(1, -1)
            distances_to_centroid = cdist(points_in_cluster, cluster_centroid_orig, 'euclidean')
            radii.append(distances_to_centroid.max())
        else:
            radii.append(0)

    # Dunn Index
    dunn = dunn_index(X, final_labels)
    dunn_values.append(dunn)

    # Вивід фінального результату
    print(f"\n=== ФІНАЛЬНИЙ РЕЗУЛЬТАТ K={K} ===")
    print(pd.DataFrame(final_centroids_orig, columns=['X1_Центроїд', 'X2_Центроїд']).round(2))
    print("Радіуси кластерів:", [f'{r:.2f}' for r in radii])
    print(f"Dunn Index: {dunn:.4f}")

    # Графік кластерів
    plot_clusters(X, final_labels, final_centroids_orig, K, radii, dunn)

# --- 6. ГРАФІК DUNN INDEX VS K ---
plt.figure(figsize=(10, 6))
plt.plot(K_values, dunn_values, marker='o', linestyle='-', color='blue')
plt.xticks(K_values)
plt.xlabel('K (кількість кластерів)')
plt.ylabel('Dunn Index')
plt.title('Dunn Index у залежності від K')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
