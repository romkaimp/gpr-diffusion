import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, Matern
from sklearn.model_selection import train_test_split
import pandas as pd


def analyze_gpr_characteristics(gpr_models, X_train, feature_names=['x', 'y', 'z']):
    """Анализ и вывод характеристик Gaussian Process моделей"""

    print("=" * 60)
    print("ХАРАКТЕРИСТИКИ GAUSSIAN PROCESS REGRESSION МОДЕЛЕЙ")
    print("=" * 60)

    results = []

    for i, (gpr, comp) in enumerate(zip(gpr_models, feature_names)):
        print(f"\n--- Компонента {comp} ---")

        # 1. Параметры ядра (до и после обучения)
        print("Ядро до обучения:", gpr.kernel)
        print("Ядро после обучения:", gpr.kernel_)

        # 2. Параметры ядра в численном виде
        kernel_params = {}
        if hasattr(gpr.kernel_, 'k1'):  # Для составных ядер
            if hasattr(gpr.kernel_.k1, 'k1') and hasattr(gpr.kernel_.k1, 'k2'):
                # RBF * Constant
                kernel_params['constant_value'] = gpr.kernel_.k1.k1.constant_value
                kernel_params['length_scale'] = gpr.kernel_.k1.k2.length_scale
                kernel_params['noise_level'] = gpr.kernel_.k2.noise_level
            else:
                # Простая структура
                kernel_params['constant_value'] = gpr.kernel_.k1.constant_value
                kernel_params['length_scale'] = gpr.kernel_.k2.length_scale
                kernel_params['noise_level'] = gpr.kernel_.k3.noise_level

        print("Параметры ядра:")
        for param, value in kernel_params.items():
            print(f"  {param}: {value}")

        # 3. Log Marginal Likelihood (LML)
        print(f"Log Marginal Likelihood: {gpr.log_marginal_likelihood():.3f}")

        # 4. Неопределенность предсказаний
        y_pred, y_std = gpr.predict(X_train[:10], return_std=True)
        avg_uncertainty = y_std.mean()
        print(f"Средняя неопределенность: {avg_uncertainty:.6f}")

        # 5. Градиенты ядра (если доступно)
        if hasattr(gpr, 'kernel_'):
            try:
                gradients = gpr.kernel_.gradient(X_train[0])
                print(f"Градиенты ядра (первые 3): {gradients[:3]}")
            except:
                print("Градиенты ядра: не доступны")

        # Сохраняем результаты
        results.append({
            'component': comp,
            'kernel_initial': str(gpr.kernel),
            'kernel_optimized': str(gpr.kernel_),
            'lml': gpr.log_marginal_likelihood(),
            'avg_uncertainty': avg_uncertainty,
            **kernel_params
        })

    return results


# Генерация данных Лоренца
def generate_lorenz_data(initial_state, num_points=2000, dt=0.01, transient=500):
    total_points = transient + num_points
    t_span = (0, total_points * dt)
    t_eval = np.linspace(0, total_points * dt, total_points)

    solution = solve_ivp(lambda t, state: [
        10.0 * (state[1] - state[0]),
        state[0] * (28.0 - state[2]) - state[1],
        state[0] * state[1] - (8.0 / 3.0) * state[2]
    ], t_span, initial_state, t_eval=t_eval, method='RK45')

    data = np.column_stack([solution.y[0][transient:],
                            solution.y[1][transient:],
                            solution.y[2][transient:]])
    return data


# Создание и обучение GPR моделей
print("Создание и обучение GPR моделей...")
initial_state = [0.0, 1.0, 1.05]
data = generate_lorenz_data(initial_state, 2000)

X = data[:-1]
y = data[1:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Разные ядра для экспериментов
kernels = [
    ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1),
    ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=0.1),
    ConstantKernel(1.0) * RBF(length_scale=[1.0, 1.0, 1.0]) + WhiteKernel(noise_level=0.1)
]

kernel_names = ['RBF', 'Matern (ν=1.5)', 'RBF с автоматическим определением масштаба']

gpr_models_list = []

for kernel, kernel_name in zip(kernels, kernel_names):
    print(f"\nОбучение с ядром: {kernel_name}")

    models = []
    for i in range(3):
        gpr = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            n_restarts_optimizer=3,
            random_state=42
        )

        # Обучаем на подвыборке
        gpr.fit(X_train[:500], y_train[:500, i])
        models.append(gpr)

    gpr_models_list.append((models, kernel_name))

# Анализ характеристик для всех моделей
all_results = []

for models, kernel_name in gpr_models_list:
    print(f"\n{'-' * 20} {kernel_name} {'-' * 20}")
    results = analyze_gpr_characteristics(models, X_train)

    for res in results:
        res['kernel_type'] = kernel_name
        all_results.append(res)

# Создаем DataFrame для удобного анализа
df_results = pd.DataFrame(all_results)
print("\n" + "=" * 60)
print("СВОДНАЯ ТАБЛИЦА ХАРАКТЕРИСТИК")
print("=" * 60)
print(df_results[['kernel_type', 'component', 'lml', 'avg_uncertainty',
                  'constant_value', 'length_scale', 'noise_level']].to_string())

# Визуализация характеристик
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Log Marginal Likelihood по компонентам и ядрам
for kernel_type in df_results['kernel_type'].unique():
    subset = df_results[df_results['kernel_type'] == kernel_type]
    axes[0, 0].bar(np.arange(3) + 0.2 * list(df_results['kernel_type'].unique()).index(kernel_type),
                   subset['lml'], width=0.2, label=kernel_type, alpha=0.8)

axes[0, 0].set_title('Log Marginal Likelihood по ядрам и компонентам')
axes[0, 0].set_xlabel('Компонента')
axes[0, 0].set_ylabel('LML')
axes[0, 0].set_xticks([0, 1, 2])
axes[0, 0].set_xticklabels(['x', 'y', 'z'])
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Параметры ядра
kernel_params = ['constant_value', 'length_scale', 'noise_level']
param_names = ['Константа', 'Масштаб длины', 'Уровень шума']

for j, param in enumerate(kernel_params):
    for i, comp in enumerate(['x', 'y', 'z']):
        subset = df_results[df_results['component'] == comp]
        values = subset[param].values
        axes[1, j // 2].bar(np.arange(len(values)) + 0.2 * i, values,
                            width=0.2, label=comp, alpha=0.8)

    axes[1, j // 2].set_title(f'Параметр: {param_names[j]}')
    axes[1, j // 2].set_xticks(np.arange(len(kernel_names)))
    axes[1, j // 2].set_xticklabels([name[:15] + '...' for name in kernel_names], rotation=45)
    axes[1, j // 2].legend()
    axes[1, j // 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gpr_characteristics_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Дополнительный анализ: ковариационная матрица
print("\nАнализ ковариационной матрицы для первой модели:")
best_models = gpr_models_list[0][0]  # Берем первую модель

for i, (gpr, comp) in enumerate(zip(best_models, ['x', 'y', 'z'])):
    print(f"\n--- Ковариационная матрица для {comp} ---")

    # Ковариация для нескольких точек
    X_sample = X_train[:5]
    K = gpr.kernel_(X_sample)

    print("Ковариационная матрица (5x5):")
    print(np.round(K, 3))

    print(f"Определитель: {np.linalg.det(K):.6f}")
    print(f"Число обусловленности: {np.linalg.cond(K):.2f}")

# Анализ чувствительности к гиперпараметрам
print("\nАнализ чувствительности LML к параметрам ядра:")

# Для первой модели и компоненты x
gpr_x = best_models[0]
original_lml = gpr_x.log_marginal_likelihood()

# Вариация length_scale
length_scales = np.linspace(0.1, 10.0, 20)
lml_values = []

for ls in length_scales:
    # Создаем копию ядра с измененным параметром
    new_kernel = gpr_x.kernel_.clone_with_theta([gpr_x.kernel_.theta[0], ls, gpr_x.kernel_.theta[2]])
    gpr_temp = GaussianProcessRegressor(kernel=new_kernel, alpha=1e-6)
    gpr_temp.fit(X_train[:100], y_train[:100, 0])  # На маленькой выборке для скорости
    lml_values.append(gpr_temp.log_marginal_likelihood())

plt.figure(figsize=(10, 6))
plt.plot(length_scales, lml_values, 'b-o', linewidth=2)
plt.axvline(gpr_x.kernel_.theta[1], color='r', linestyle='--',
            label=f'Оптимальное значение: {gpr_x.kernel_.theta[1]:.3f}')
plt.xlabel('Масштаб длины (length_scale)')
plt.ylabel('Log Marginal Likelihood')
plt.title('Чувствительность LML к параметру length_scale')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('gpr_lml_sensitivity.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nАнализ завершен! Все характеристики сохранены.")