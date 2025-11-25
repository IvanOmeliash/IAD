import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from matplotlib.patches import Circle
import time

# --- 1. ПІДГОТОВКА ДАНИХ ТА НАЛАШТУВАННЯ ---

# Вхідні дані
data = {
    'X1': [5, 4, 6, 80, 95, 70, 40, 50, 10, 15, 90, 20, 12, 100],
    'X2': [150, 140, 160, 350, 400, 320, 250, 280, 200, 180, 380, 200, 170, 450]
}
df = pd.DataFrame(data)
X = df[['X1', 'X2']].values  # Дані як масив NumPy

# Параметри K-Means
K = 3
MAX_ITER = 10
np.random.seed(42)  # Фіксуємо початкові центроїди для відтворюваності

final_centroids_scaled = None
final_labels = None

# Стандартизація даних
# Це обов'язково, щоб ознаки X1 та X2 мали однаковий вплив
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Ініціалізація центроїдів: випадковий вибір K точок
# Використовуємо перші K точок для простоти демонстрації
initial_indices = np.random.choice(len(X_scaled), K, replace=False)  # Випадковий вибір K унікальних індексів
centroids = X_scaled[initial_indices]

print(f"--- 💡 Початкова Ініціалізація (K={K}) ---")
print("Початкові центроїди (в оригінальному масштабі):\n",
      pd.DataFrame(scaler.inverse_transform(centroids),
                   columns=['X1_Центроїд', 'X2_Центроїд']).round(1))


# --- 2. ФУНКЦІЇ ДЛЯ КРОКІВ АЛГОРИТМУ ---

def plot_iteration(X_orig, X_scaled, centroids_scaled, iteration, scaler):
    """Візуалізує поточний стан кластеризації."""

    # 1. Призначення кластерів
    distances = cdist(X_scaled, centroids_scaled, 'euclidean')
    labels = np.argmin(distances, axis=1)

    # 2. Зворотне перетворення центроїдів для підпису
    centroids_orig_scale = scaler.inverse_transform(centroids_scaled)

    # Побудова графіка
    plt.figure(figsize=(10, 7))

    # Точки даних, пофарбовані за кластером
    scatter = plt.scatter(X_orig[:, 0], X_orig[:, 1], c=labels,
                          cmap='viridis', s=100, alpha=0.8, edgecolor='k')

    # Центроїди: позначаються червоними 'X'
    plt.scatter(centroids_orig_scale[:, 0], centroids_orig_scale[:, 1],
                marker='X', s=250, c='red', label='Центроїди', edgecolor='k', linewidths=2)

    title_str = (f'K-Means Ітерація {iteration} (K={K})\n'
                 f'Ц0:{centroids_orig_scale[0].round(1)}, '
                 f'Ц1:{centroids_orig_scale[1].round(1)}, '
                 f'Ц2:{centroids_orig_scale[2].round(1)}')

    plt.title(title_str)
    plt.xlabel('Ознака 1: Кількість запитів (X1, запитів/сек)')
    plt.ylabel('Ознака 2: Середній час відгуку (X2, мс)')

    legend1 = plt.legend(*scatter.legend_elements(), title="Кластери", loc="upper left")
    plt.gca().add_artist(legend1)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.show() # Розкоментуйте, якщо хочете бачити графіки в інтерактивному режимі


# --- 3. ПОКРОКОВЕ ВИКОНАННЯ K-MEANS ---

for i in range(1, MAX_ITER + 1):
    print(f"\n\n==================== 🔄 ІТЕРАЦІЯ {i} ====================")

    # --- КРОК E: ПРИЗНАЧЕННЯ КЛАСТЕРІВ (Розрахунок відстаней) ---

    # Обчислюємо Евклідову відстань від кожної точки до кожного центроїда
    distances = cdist(X_scaled, centroids, 'euclidean')
    labels = np.argmin(distances, axis=1)  # Індекс найближчого центроїда (кластер)

    # Таблиця відстаней
    distances_df = pd.DataFrame(distances, columns=[f'Відстань до Ц{j}' for j in range(K)])
    distances_df['Кластер'] = labels

    print(f"--- 1. Таблиця Відстаней (На основі центроїдів Ітерації {i - 1}) ---")
    print(distances_df.head(14).to_string())  # Вивід всіх 14 точок

    # --- КРОК M: ОНОВЛЕННЯ ЦЕНТРОЇДІВ ---

    new_centroids = np.zeros(centroids.shape)
    converged = True

    for k in range(K):
        # Вибираємо всі точки, що належать кластеру k
        points_in_cluster = X_scaled[labels == k]

        if len(points_in_cluster) > 0:
            # Обчислюємо нове середнє значення
            new_centroids[k] = points_in_cluster.mean(axis=0)
        else:
            # Якщо кластер порожній, центроїд не змінюється
            new_centroids[k] = centroids[k]

    # --- ПЕРЕВІРКА НА ЗБІЖНІСТЬ ---
    if np.allclose(centroids, new_centroids):
        converged = True

        final_centroids_scaled = new_centroids
        final_labels = labels
    else:
        converged = False

    # --- ВИВЕДЕННЯ НОВИХ ЦЕНТРОЇДІВ ---
    centroids = new_centroids
    centroids_orig = scaler.inverse_transform(centroids)
    centroids_df_iter = pd.DataFrame(centroids_orig,
                                     columns=['X1_Центроїд', 'X2_Центроїд']).round(4)

    print(f"\n--- 2. Нові Координати Центроїдів (Оригінальний Масштаб) ---")
    print(centroids_df_iter.to_string())

    # --- ВІЗУАЛІЗАЦІЯ ---
    plot_iteration(X, X_scaled, centroids, i, scaler)

    if converged:
        print(f"\n--- ✅ ЗБІЖНІСТЬ ДОСЯГНУТА ---")
        print(f"Алгоритм зійшовся на Ітерації {i}. Фінальний результат отримано.")
        break

# --- 4. ФІНАЛЬНА ВІЗУАЛІЗАЦІЯ З РАДІУСАМИ ---
if final_centroids_scaled is not None:
    print("\n\n==================== 👑 ФІНАЛЬНИЙ РЕЗУЛЬТАТ З РАДІУСАМИ ====================")

    df['Кластер'] = final_labels
    final_centroids_orig = scaler.inverse_transform(final_centroids_scaled)
    radii = []

    # Обчислення Радіуса Кластера
    for k in range(K):
        # Точки в оригінальному масштабі, що належать кластеру k
        points_in_cluster = X[final_labels == k]

        if len(points_in_cluster) > 0:
            cluster_centroid_orig = final_centroids_orig[k].reshape(1, -1)
            # Обчислюємо Евклідову відстань від центроїда до всіх точок кластера
            distances_to_centroid = cdist(points_in_cluster, cluster_centroid_orig, 'euclidean')
            max_distance = distances_to_centroid.max()
            radii.append(max_distance)
        else:
            radii.append(0)

    print("Радіуси кластерів (макс. відстань від центру до точки):", [f'{r:.2f}' for r in radii])

    # --- Візуалізація ---
    plt.figure(figsize=(12, 8))
    colors = ['purple', 'blue', 'green']

    # A. Додавання Кілець (Радіусів)
    for k in range(K):
        if radii[k] > 0:
            circle = Circle(final_centroids_orig[k], radii[k],
                            color=colors[k], fill=False, linestyle='--',
                            linewidth=1.5, alpha=0.6)
            plt.gca().add_patch(circle)

    # B. Діаграма розсіювання: точки даних
    scatter = plt.scatter(df['X1'], df['X2'], c=df['Кластер'],
                          cmap='viridis', s=100, alpha=0.8, edgecolor='k')

    # C. Додавання Центроїдів
    plt.scatter(final_centroids_orig[:, 0], final_centroids_orig[:, 1],
                marker='X', s=300, c='red', label='Центроїди', edgecolor='black', linewidths=2)

    # D. Налаштування графіку
    plt.title('K-Means: Фінальний результат з Кластерними Радіусами (K=3)')
    plt.xlabel('Ознака 1: Кількість запитів (X1, запитів/сек)')
    plt.ylabel('Ознака 2: Середній час відгуку (X2, мс)')

    # Легенда
    plt.legend(loc='lower right', handles=[
        plt.Line2D([0], [0], marker='X', color='w', label='Центроїд', markerfacecolor='red', markersize=15),
        plt.Line2D([0], [0], color=colors[0], linestyle='--', linewidth=1.5, label=f'Радіус Ц0: {radii[0]:.2f}'),
        plt.Line2D([0], [0], color=colors[1], linestyle='--', linewidth=1.5, label=f'Радіус Ц1: {radii[1]:.2f}'),
        plt.Line2D([0], [0], color=colors[2], linestyle='--', linewidth=1.5, label=f'Радіус Ц2: {radii[2]:.2f}')
    ])
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

# ---

if not converged:
    print(f"\n--- ⚠️ МАКСИМАЛЬНА КІЛЬКІСТЬ ІТЕРАЦІЙ ДОСЯГНУТА ---")