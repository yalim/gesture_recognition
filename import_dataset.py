import numpy as np


def load_data(filename):
    return np.genfromtxt('./'+filename+'.csv', delimiter=',')

# Import the dataset
y = load_data('labels')
user_ids = np.asarray(map(int, load_data('user_ids')))
x_accel = load_data('x_acc_noise').reshape((y.shape[0], 100, 1))
y_accel = load_data('y_acc_noise').reshape((y.shape[0], 100, 1))
z_accel = load_data('z_acc_noise').reshape((y.shape[0], 100, 1))
x_gyro = load_data('x_gyr_noise').reshape((y.shape[0], 100, 1))
y_gyro = load_data('y_gyr_noise').reshape((y.shape[0], 100, 1))
z_gyro = load_data('z_gyr_noise').reshape((y.shape[0], 100, 1))

# Choose users to be in validation, test and training sets
val_cond = [(6 == user_ids) | (3 == user_ids)]
test_cond = [(4 == user_ids) | (2 == user_ids)]
train_cond = [(2 != user_ids) & (3 != user_ids) & (4 != user_ids) & (6 != user_ids)]

x_accel_test = x_accel[test_cond]
x_accel_validation = x_accel[val_cond]
x_accel_train = x_accel[train_cond]

y_accel_test = y_accel[test_cond]
y_accel_validation = y_accel[val_cond]
y_accel_train = y_accel[train_cond]

z_accel_test = z_accel[test_cond]
z_accel_validation = z_accel[val_cond]
z_accel_train = z_accel[train_cond]

x_gyro_test = x_gyro[test_cond]
x_gyro_validation = x_gyro[val_cond]
x_gyro_train = x_gyro[train_cond]

y_gyro_test = y_gyro[test_cond]
y_gyro_validation = y_gyro[val_cond]
y_gyro_train = y_gyro[train_cond]

z_gyro_test = z_gyro[test_cond]
z_gyro_validation = z_gyro[val_cond]
z_gyro_train = z_gyro[train_cond]


y_test = y[test_cond]
y_validation = y[val_cond]
y_train = y[train_cond]
