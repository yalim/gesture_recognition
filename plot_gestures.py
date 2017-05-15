"""Plots the histogram of all gestures. Also it plots each gesture individually."""

# import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def load_data(filename):
    names = []
    for i in range(100):
        names.append(i)
    return pd.read_csv('./' + filename + '.csv', names=names)

# Import the dataset
x_accel = load_data('x_acc_raw')
y_accel = load_data('y_acc_raw')
z_accel = load_data('z_acc_raw')
x_gyro = load_data('x_gyr_raw')
y_gyro = load_data('y_gyr_raw')
z_gyro = load_data('z_gyr_raw')

plt.figure(1)
plt.hist(x_accel)
plt.title('x_accel')

plt.figure(2)
plt.hist(y_accel)
plt.title('y_accel')

plt.figure(3)
plt.hist(z_accel)
plt.title('z_accel')

plt.figure(4)
plt.hist(x_gyro)
plt.title('x_gyro')

plt.figure(5)
plt.hist(y_gyro)
plt.title('y_gyro')

plt.figure(6)
plt.hist(z_gyro)
plt.title('z_gyro')

plt.show()
