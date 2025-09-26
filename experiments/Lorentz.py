import numpy as np

train_data = np.load('lorenz_train.npy')
test_data = np.load('lorenz_test.npy')

# Для прогнозирования временных рядов:
# X_train: [x_t, y_t, z_t] → y_train: [x_{t+1}, y_{t+1}, z_{t+1}]
X_train = train_data[:-1]  # все кроме последней точки
y_train = train_data[1:]   # все кроме первой точки

# Аналогично для тестовых данных
X_test = test_data[:-1]
y_test = test_data[1:]