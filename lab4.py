import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform

# --- 1. Вхідні дані ---
data = np.array([
    [5, 150], [4, 140], [6, 160], [80, 350], [95, 400], [70, 320], [40, 250],
    [50, 280], [15, 180], [90, 380], [20, 200], [12, 170], [100, 450]
])

labels = [f'№{i+1}' for i in range(len(data))]

# --- 2а. Матриця попарних відстаней ---
dist_matrix = squareform(pdist(data, metric='euclidean'))
df_dist = pd.DataFrame(dist_matrix, columns=labels, index=labels)
print("\n--- Матриця евклідових відстаней ---")
print(df_dist.round(2).to_string())

# --- 2. Ієрархічна кластеризація ---
Z = linkage(data, method='complete', metric='euclidean')

# --- 3. Дендрограма ---
plt.figure(figsize=(12, 7))
dendrogram(Z, labels=labels, orientation='top', distance_sort='descending', color_threshold=200)
plt.title('Дендрограма ієрархічної кластеризації (Complete Linkage)')
plt.xlabel('Об\'єкти')
plt.ylabel('Відстань об\'єднання')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# --- 4. Матриця об'єднань ---
df_linkage = pd.DataFrame(Z, columns=[
    'Об\'єкт A (індекс)', 'Об\'єкт B (індекс)', 'Відстань об\'єднання', 'Розмір кластера'
])

def format_index(idx):
    if idx < len(data):
        return f'№{int(idx) + 1}'
    else:
        return f'Кластер {int(idx) + 1 - len(data)}'

df_linkage['Об\'єкт A'] = df_linkage['Об\'єкт A (індекс)'].apply(format_index)
df_linkage['Об\'єкт B'] = df_linkage['Об\'єкт B (індекс)'].apply(format_index)
print(df_linkage[['Об\'єкт A', 'Об\'єкт B', 'Відстань об\'єднання', 'Розмір кластера']]
      .to_string(index=False, float_format='{:.2f}'.format))

# --- 5. Візуалізація крок за кроком зі стрілкою між об'єктами ---
def get_clusters(Z, step, n_points):
    clusters = {i: [i] for i in range(n_points)}
    next_cluster_id = n_points
    for i in range(step):
        a, b = int(Z[i, 0]), int(Z[i, 1])
        cluster_a = clusters.pop(a)
        cluster_b = clusters.pop(b)
        clusters[next_cluster_id] = cluster_a + cluster_b
        next_cluster_id += 1
    return clusters

plt.figure(figsize=(8, 7))

for step in range(1, len(Z) + 1):
    plt.clf()
    plt.title(f"Крок {step}: формування кластерів")
    plt.xlabel("X1")
    plt.ylabel("X2")

    clusters = get_clusters(Z, step, len(data))

    # масштаб
    plt.xlim(data[:, 0].min() - 20, data[:, 0].max() + 20)
    plt.ylim(data[:, 1].min() - 50, data[:, 1].max() + 50)

    # малюємо точки
    colors = cm.tab20(np.linspace(0, 1, len(clusters)))
    for color, cluster in zip(colors, clusters.values()):
        pts = data[cluster]
        plt.scatter(pts[:, 0], pts[:, 1], s=80, color=color)

    # --- знаходимо конкретні точки для стрілки ---
    a_idx, b_idx = int(Z[step - 1, 0]), int(Z[step - 1, 1])
    prev_clusters = get_clusters(Z, step - 1, len(data))
    cluster_a_points = data[prev_clusters[a_idx]]
    cluster_b_points = data[prev_clusters[b_idx]]

    # complete linkage -> максимальна відстань між кластерами
    max_dist = 0
    point_a, point_b = None, None
    for pa in cluster_a_points:
        for pb in cluster_b_points:
            d = np.linalg.norm(pa - pb)
            if d > max_dist:
                max_dist = d
                point_a, point_b = pa, pb

    # стрілка від конкретного об'єкта до іншого
    plt.arrow(
        point_a[0], point_a[1],
        point_b[0] - point_a[0],
        point_b[1] - point_a[1],
        width=0.8,
        head_width=3,
        length_includes_head=True,
        color='black'
    )

    plt.pause(13.0)

plt.show()
