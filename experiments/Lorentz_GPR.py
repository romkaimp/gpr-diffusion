import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time


def lorenz_system(t, state, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    """Система уравнений Лоренца"""
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return np.array([dx_dt, dy_dt, dz_dt])


def generate_lorenz_data(initial_state, num_points=10000, dt=0.01, transient=1000):
    """Генерация данных системы Лоренца"""
    total_points = transient + num_points
    t_span = (0, total_points * dt)
    t_eval = np.linspace(0, total_points * dt, total_points)

    solution = solve_ivp(lorenz_system, t_span, initial_state,
                         t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-9)

    # Отбрасываем переходный процесс
    x = solution.y[0][transient:]
    y = solution.y[1][transient:]
    z = solution.y[2][transient:]

    return np.column_stack([x, y, z])


# Параметры
initial_state = [0.0, 1.0, 1.05]
num_train = 2000  # Уменьшаем для скорости обучения
num_test = 1000
dt = 0.01

print("Генерация данных Лоренца...")
data = generate_lorenz_data(initial_state, num_train + num_test, dt)

# Подготовка данных для обучения
X = data[:-1]  # Текущее состояние
y = data[1:]  # Следующее состояние

# Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=num_test / (num_train + num_test), random_state=42, shuffle=False
)

print(f"Размеры данных: Train {X_train.shape}, Test {X_test.shape}")

# Создание и обучение GPR моделей для каждой компоненты
print("\nОбучение GPR моделей...")

# Ядро для Gaussian Process
kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)

# Создаем отдельные модели для каждой компоненты
gpr_models = []
predictions = []

start_time = time.time()

for i in range(3):  # Для x, y, z
    print(f"Обучение модели для компоненты {['x', 'y', 'z'][i]}...")

    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,  # Небольшой шум для стабильности
        n_restarts_optimizer=5,
        random_state=42
    )

    # Обучаем на подвыборке для скорости (можно увеличить для лучшей точности)
    subset_size = min(1000, len(X_train))
    indices = np.random.choice(len(X_train), subset_size, replace=False)

    gpr.fit(X_train[indices], y_train[indices, i])
    gpr_models.append(gpr)

    # Предсказание на тестовых данных
    y_pred, y_std = gpr.predict(X_test, return_std=True)
    predictions.append((y_pred, y_std))

    print(f"  Компонента {['x', 'y', 'z'][i]} - обучена")

training_time = time.time() - start_time
print(f"\nОбучение завершено за {training_time:.2f} секунд")

# Анализ результатов
print("\nОценка качества предсказаний:")

for i, comp in enumerate(['x', 'y', 'z']):
    y_pred, y_std = predictions[i]
    mse = mean_squared_error(y_test[:, i], y_pred)
    mae = mean_absolute_error(y_test[:, i], y_pred)

    print(f"{comp}: MSE = {mse:.6f}, MAE = {mae:.6f}")
    print(f"  Средняя неопределенность: {y_std.mean():.6f}")

# Визуализация предсказаний
fig, axes = plt.subplots(3, 2, figsize=(15, 12))

for i, comp in enumerate(['x', 'y', 'z']):
    y_pred, y_std = predictions[i]

    # Временные ряды
    axes[i, 0].plot(y_test[:200, i], 'b-', label='Истинные значения', alpha=0.8)
    axes[i, 0].plot(y_pred[:200], 'r--', label='GPR предсказания', alpha=0.8)
    axes[i, 0].fill_between(range(200),
                            y_pred[:200] - 2 * y_std[:200],
                            y_pred[:200] + 2 * y_std[:200],
                            alpha=0.3, color='red', label='±2σ')
    axes[i, 0].set_title(f'Компонента {comp} - Предсказания vs Истинные значения')
    axes[i, 0].set_xlabel('Временной шаг')
    axes[i, 0].set_ylabel('Значение')
    axes[i, 0].legend()
    axes[i, 0].grid(True, alpha=0.3)

    # Диаграмма рассеяния
    axes[i, 1].scatter(y_test[:, i], y_pred, alpha=0.6, s=10)
    axes[i, 1].plot([y_test[:, i].min(), y_test[:, i].max()],
                    [y_test[:, i].min(), y_test[:, i].max()],
                    'r--', lw=2)
    axes[i, 1].set_title(f'Компонента {comp} - Диаграмма рассеяния')
    axes[i, 1].set_xlabel('Истинные значения')
    axes[i, 1].set_ylabel('Предсказания')
    axes[i, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gpr_lorenz_predictions.png', dpi=300, bbox_inches='tight')
plt.show()

# 3D визуализация истинной и предсказанной траекторий
fig = plt.figure(figsize=(15, 6))

# Истинная траектория
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(y_test[:500, 0], y_test[:500, 1], y_test[:500, 2],
         'b-', lw=0.5, alpha=0.8, label='Истинная')
ax1.set_title('Истинная траектория Лоренца')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

# Предсказанная траектория
ax2 = fig.add_subplot(122, projection='3d')
pred_trajectory = np.column_stack([predictions[0][0], predictions[1][0], predictions[2][0]])
ax2.plot(pred_trajectory[:500, 0], pred_trajectory[:500, 1], pred_trajectory[:500, 2],
         'r-', lw=0.5, alpha=0.8, label='GPR предсказание')
ax2.set_title('GPR предсказанная траектория')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')

plt.tight_layout()
plt.savefig('gpr_lorenz_3d_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Долгосрочное предсказание (итеративное)
print("\nЗапуск итеративного предсказания...")


def predict_long_term(initial_state, gpr_models, steps=1000):
    """Итеративное предсказание на много шагов вперед"""
    current_state = initial_state.copy()
    trajectory = [current_state]

    for step in range(steps):
        # Предсказываем следующее состояние для каждой компоненты
        next_state = []
        for i, gpr in enumerate(gpr_models):
            pred, std = gpr.predict([current_state], return_std=True)
            next_state.append(pred[0])

        current_state = np.array(next_state)
        trajectory.append(current_state)

    return np.array(trajectory)


# Запускаем долгосрочное предсказание
long_term_pred = predict_long_term(X_test[0], gpr_models, steps=500)

# Сравнение с истинной траекторией
fig, axes = plt.subplots(3, 1, figsize=(12, 9))

for i, comp in enumerate(['x', 'y', 'z']):
    axes[i].plot(y_test[:500, i], 'b-', label='Истинная', alpha=0.8)
    axes[i].plot(long_term_pred[:500, i], 'r--', label='GPR долгосрочное', alpha=0.8)
    axes[i].set_title(f'Долгосрочное предсказание - {comp}')
    axes[i].set_xlabel('Временной шаг')
    axes[i].set_ylabel('Значение')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gpr_lorenz_long_term.png', dpi=300, bbox_inches='tight')
plt.show()

print("Анализ завершен! Все графики сохранены.")