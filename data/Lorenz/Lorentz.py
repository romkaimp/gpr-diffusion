import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd
import random
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Важно для 3D проекции


def generate_lorenz_data_reconstruction1(sigma, rho, beta, trajectory_size=10000,
                         num_trajectories=100, seq_length=40, dt=0.01, transient=1000):
    '''
    :param sigma: float Параметры системы Лоренца
    :param rho:
    :param beta:
    :param trajectory_size:
    :param num_trajectories:
    :param seq_length:
    :param dt: Шаг времени для дискретизации
    :param transient: Количество начальных точек для отбрасывания (переходный процесс)
    :return: train_data : ndarray
        Тренировочные данные формы (train_size, 3)
    '''
    # Общее количество точек для генерации (+ переходный процесс)
    total_points = transient + trajectory_size

    # Временной интервал
    t_span = (0, total_points * dt)
    t_eval = np.linspace(0, total_points * dt, total_points)

    dataset = []

    for i in range(num_trajectories):
        # Решаем систему уравнений
        initial_state = [random.uniform(-20, 20), random.uniform(-25, 25), random.uniform(5, 40)]
        solution = solve_ivp(
            lambda t, state: lorenz_system(state, sigma, rho, beta),
            t_span, initial_state, t_eval=t_eval,
            method='RK45', rtol=1e-6, atol=1e-9
        )

        # Извлекаем результаты
        x = solution.y[0]
        y = solution.y[1]
        z = solution.y[2]

        # Отбрасываем переходный процесс
        x = x[transient:]
        y = y[transient:]
        z = z[transient:]

        # Формируем матрицу данных [x, y, z]
        data = np.column_stack([x, y, z])
        num_sequences = (trajectory_size - seq_length) // seq_length

        for j in range(num_sequences):
            start_idx = j * seq_length
            end_idx = start_idx + seq_length
            sequence = data[start_idx:end_idx]
            dataset.append(sequence)

        # Преобразуем в numpy array
    dataset = np.array(dataset)

    # Перемешиваем данные
    np.random.shuffle(dataset)

    return dataset

def generate_lorenz_data_reconstruction(
    sigma, rho, beta, trajectory_size=10000,
    num_trajectories=100, seq_length=40, dt=0.01, transient=1000,
    sliding=True
):
    '''
    :param sigma: float Параметры системы Лоренца
    :param rho:
    :param beta:
    :param trajectory_size: количество точек после отбрасывания transient
    :param num_trajectories: количество траекторий
    :param seq_length: длина последовательности (окна)
    :param dt: шаг интегрирования
    :param transient: количество точек для "разгона"
    :param sliding: если True — использовать скользящее окно, иначе дискретные окна
    :return: dataset : ndarray
        Данные формы (N, seq_length, 3)
    '''
    total_points = transient + trajectory_size

    # Временной интервал
    t_span = (0, total_points * dt)
    t_eval = np.linspace(0, total_points * dt, total_points)

    dataset = []
    for i in range(num_trajectories):
        # Случайные начальные условия
        initial_state = [
            random.uniform(-20, 20),
            random.uniform(-25, 25),
            random.uniform(5, 40)
        ]

        solution = solve_ivp(
            lambda t, state: lorenz_system(state, sigma, rho, beta),
            t_span, initial_state, t_eval=t_eval,
            method='RK45', rtol=1e-6, atol=1e-9
        )

        # Извлекаем результаты
        x, y, z = solution.y
        # Отбрасываем переходный процесс
        data = np.column_stack([x[transient:], y[transient:], z[transient:]])

        if sliding:
            # Скользящее окно
            num_sequences = trajectory_size - seq_length + 1
            for j in range(0, num_sequences, 64):
                dataset.append(data[j:j+seq_length])
        else:
            # Непересекающиеся окна
            num_sequences = (trajectory_size - seq_length) // seq_length
            for j in range(num_sequences):
                start_idx = j * seq_length
                end_idx = start_idx + seq_length
                dataset.append(data[start_idx:end_idx])

    dataset = np.array(dataset)
    np.random.shuffle(dataset)
    return dataset

def generate_lorenz_data_prediction_1(sigma, rho, beta, trajectory_size=10000,
                         num_trajectories=100, seq_length=32, test_length=32, dt=0.02, transient=1000):
    '''

    :param sigma:
    :param rho:
    :param beta:
    :param trajectory_size:
    :param num_trajectories:
    :param seq_length:
    :param test_length:
    :param dt:
    :param transient:
    :return:
    '''
    total_points = transient + trajectory_size

    # Временной интервал
    t_span = (0, total_points * dt)
    t_eval = np.linspace(0, total_points * dt, total_points)

    dataset = []
    dataset_test = []

    for i in range(num_trajectories):
        # Решаем систему уравнений
        initial_state = [random.uniform(-20, 20), random.uniform(-25, 25), random.uniform(5, 40)]
        solution = solve_ivp(
            lambda t, state: lorenz_system(state, sigma, rho, beta),
            t_span, initial_state, t_eval=t_eval,
            method='RK45', rtol=1e-6, atol=1e-9
        )

        # Извлекаем результаты
        x = solution.y[0]
        y = solution.y[1]
        z = solution.y[2]

        # Отбрасываем переходный процесс
        x = x[transient:]
        y = y[transient:]
        z = z[transient:]

        # Формируем матрицу данных [x, y, z]
        data = np.column_stack([x, y, z])
        num_sequences = (trajectory_size - seq_length - test_length) // (seq_length + test_length)

        for j in range(num_sequences):
            start_idx = j * (seq_length + test_length)
            end_idx = start_idx + (seq_length + test_length)
            sequence = data[start_idx:start_idx+seq_length]
            test = data[start_idx+seq_length:end_idx]
            dataset.append(sequence)
            dataset_test.append(test)

        # Преобразуем в numpy array
    dataset = np.array(dataset)
    dataset_test = np.array(dataset_test)

    return dataset, dataset_test

def generate_lorenz_data_prediction(
        sigma, rho, beta,
        trajectory_size=10000,
        num_trajectories=100,
        seq_length=32,
        test_length=32,
        dt=0.02, transient=1000,
        sliding_window=True
    ):
    """
    Генерация датасета для задачи предсказания:
    X (train) -> Y (target future).
    """

    total_points = transient + trajectory_size
    t_span = (0, total_points * dt)
    t_eval = np.linspace(0, total_points * dt, total_points)

    X, Y = [], []

    for i in range(num_trajectories):
        # случайные начальные условия
        initial_state = [
            random.uniform(-20, 20),
            random.uniform(-25, 25),
            random.uniform(5, 40)
        ]

        solution = solve_ivp(
            lambda t, state: lorenz_system(state, sigma, rho, beta),
            t_span, initial_state, t_eval=t_eval,
            method="RK45", rtol=1e-6, atol=1e-9
        )

        # данные после трансита
        data = np.column_stack([solution.y[0], solution.y[1], solution.y[2]])
        data = data[transient:]  # shape: (trajectory_size, 3)

        if sliding_window:
            # скользящее окно
            for start in range(0, trajectory_size - seq_length - test_length, 64):
                X.append(data[start:start+seq_length])
                Y.append(data[start+seq_length:start+seq_length+test_length])
        else:
            # дискретные шаги без пересечений
            num_sequences = (trajectory_size - seq_length - test_length) // (seq_length + test_length)
            for j in range(num_sequences):
                start = j * (seq_length + test_length)
                X.append(data[start:start+seq_length])
                Y.append(data[start+seq_length:start+seq_length+test_length])

    return np.array(X), np.array(Y)

def generate_single_lorenz_trajectory(initial_state, sigma=10.0, rho=28.0, beta=8.0 / 3.0,
                                      num_points=10000, dt=0.01, transient=1000):
    """
    Returns:
    --------
    data : ndarray
        Массив формы (num_points, 3) с координатами [x, y, z]
    """

    # Общее количество точек для генерации
    total_points = transient + num_points
    t_span = (0, total_points * dt)
    t_eval = np.linspace(0, total_points * dt, total_points)

    # Решаем систему уравнений
    solution = solve_ivp(
        lambda t, state: lorenz_system(state, sigma, rho, beta),
        t_span, initial_state, t_eval=t_eval,
        method='RK45', rtol=1e-6, atol=1e-9
    )

    # Извлекаем результаты и отбрасываем переходный процесс
    x = solution.y[0][transient:]
    y = solution.y[1][transient:]
    z = solution.y[2][transient:]

    # Формируем матрицу данных [x, y, z]
    data = np.column_stack([x, y, z])

    return data

def lorenz_system(state, sigma, rho, beta):
    """Система уравнений Лоренца"""
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]

def lorenz_visualize(trajectory):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
            lw=0.3, color='crimson', alpha=0.8)
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title('Отдельная траектория аттрактора Лоренца')
    plt.savefig('lorenz_single_trajectory.png', dpi=300, bbox_inches='tight')
    plt.show()

# Параметры системы (классические значения Лоренца)
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0
initial_state = [0.0, 1.0, 1.05]

def main():
    parser = argparse.ArgumentParser(description="CLI для генерации данных Лоренца")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Subcommand: single trajectory ---
    parser_single = subparsers.add_parser("single", help="Генерация одной траектории")
    parser_single.add_argument("x0", type=float, help="Начальное x")
    parser_single.add_argument("y0", type=float, help="Начальное y")
    parser_single.add_argument("z0", type=float, help="Начальное z")
    parser_single.add_argument("--sigma", type=float, default=10.0)
    parser_single.add_argument("--rho", type=float, default=28.0)
    parser_single.add_argument("--beta", type=float, default=8.0/3.0)
    parser_single.add_argument("--num_points", type=int, default=10000)
    parser_single.add_argument("--dt", type=float, default=0.01)
    parser_single.add_argument("--transient", type=int, default=1000)
    parser_single.add_argument("--output", type=str, default="single_lorenz.npy")
    parser_single.add_argument("--plot", action="store_true", help="Визуализировать и сохранить 3D траекторию")

    # --- Subcommand: reconstruction ---
    parser_recon = subparsers.add_parser("reconstruct", help="Генерация множества траекторий для задач реконструкции")
    parser_recon.add_argument("sigma", type=float)
    parser_recon.add_argument("rho", type=float)
    parser_recon.add_argument("beta", type=float)
    parser_recon.add_argument("--trajectory_size", type=int, default=10000)
    parser_recon.add_argument("--num_trajectories", type=int, default=100)
    parser_recon.add_argument("--seq_length", type=int, default=256)
    parser_recon.add_argument("--dt", type=float, default=0.02)
    parser_recon.add_argument("--transient", type=int, default=1000)
    parser_recon.add_argument("--output", type=str, default="lorenz_data_rec.npy")

    # --- Subcommand: prediction ---
    parser_recon = subparsers.add_parser("predict", help="Генерация множества траекторий для задач предсказаний")
    parser_recon.add_argument("sigma", type=float)
    parser_recon.add_argument("rho", type=float)
    parser_recon.add_argument("beta", type=float)
    parser_recon.add_argument("--trajectory_size", type=int, default=10000)
    parser_recon.add_argument("--num_trajectories", type=int, default=300)
    parser_recon.add_argument("--seq_length", type=int, default=256)
    parser_recon.add_argument("--test_length", type=int, default=2048)
    parser_recon.add_argument("--dt", type=float, default=0.02)
    parser_recon.add_argument("--transient", type=int, default=1000)
    parser_recon.add_argument("--output_train", type=str, default="lorenz_data_predict_train.npy")
    parser_recon.add_argument("--output_test", type=str, default="lorenz_data_predict_test.npy")

    args = parser.parse_args()

    if args.command == "single":
        init = [args.x0, args.y0, args.z0]
        data = generate_single_lorenz_trajectory(
            init,
            sigma=args.sigma, rho=args.rho, beta=args.beta,
            num_points=args.num_points, dt=args.dt, transient=args.transient
        )
        np.save(args.output, data)
        print(f"[SINGLE] Данные сохранены в {args.output}, shape={data.shape}")

        if args.plot:
            print("Визуализация траектории...")
            lorenz_visualize(data)

    elif args.command == "reconstruct":
        data = generate_lorenz_data_reconstruction(
            sigma=args.sigma, rho=args.rho, beta=args.beta,
            trajectory_size=args.trajectory_size,
            num_trajectories=args.num_trajectories,
            seq_length=args.seq_length,
            dt=args.dt, transient=args.transient
        )
        name = (args.output[:-4] + str(args.sigma).replace(".", "_") + "," +
        str(args.rho).replace(".", "_") + "," +
        str(args.beta).replace(".", "_") + ".npy")
        np.save(name, data)
        print(f"[RECONSTRUCT] Данные сохранены в {args.output}, shape={data.shape}")

    elif args.command == "predict":
        data, data_y = generate_lorenz_data_prediction(
            sigma=args.sigma, rho=args.rho, beta=args.beta,
            trajectory_size=args.trajectory_size,
            num_trajectories=args.num_trajectories,
            seq_length=args.seq_length,
            test_length=args.test_length,
            dt=args.dt, transient=args.transient
        )

        name_train = (args.output_train[:-4] + str(args.sigma).replace(".", "_") + "," +
                      str(args.rho).replace(".", "_") + "," +
                      str(args.beta).replace(".", "_") + ".npy")
        np.save(name_train, data)

        name_test = (args.output_test[:-4] + str(args.sigma).replace(".", "_") + "," +
                     str(args.rho).replace(".", "_") + "," +
                     str(args.beta).replace(".", "_") + ".npy")
        np.save(name_test, data_y)
        print(f"[PREDICT] Данные сохранены в {args.output_train}, shape={data.shape}, {data_y.shape}")

if __name__ == "__main__":
    #default: python .\Lorentz.py reconstruct 0.1 0.0 1.1
    #val: python .\Lorentz.py reconstruct 10 28 2.667 --output lorenz_data_rec_test.npy --trajectory_size 4000

    #default: python .\Lorentz.py single 10 28 2.667 --num_point 5000 --plot
    #default: python .\Lorentz.py predict 10 28 2.667
    #val: python .\Lorentz.py predict 10 28 2.667 --num_trajectories 100 --trajectory_size 6000 --output_train lorenz_data_predict_train_val.npy --output_test lorenz_data_predict_test_val.npy
    main()