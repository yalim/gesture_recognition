"""Import csv. and form dataset"""
import csv
import numpy as np
import matplotlib.pyplot as plt


# File location and file names
folder = './datasets_mobile/'

ccw_circle_file   = folder + 'ccw_circle.csv'
cw_circle_file    = folder + 'cw_circle.csv'
jerk_up_file      = folder + 'jerk_up.csv'
jerk_down_file    = folder + 'jerk_down.csv'
jerk_right_file   = folder + 'jerk_right.csv'
jerk_left_file    = folder + 'jerk_left.csv'
ccw_triangle_file = folder + 'ccw_triangle.csv'
cw_triangle_file  = folder + 'cw_triangle.csv'
zorro_file        = folder + 'zorro.csv'

files = [ccw_circle_file, cw_circle_file, jerk_up_file,
jerk_down_file, jerk_right_file, jerk_left_file,
ccw_triangle_file, cw_triangle_file, zorro_file]

gesture_list = ['gest_ccw_circle', 'gest_cw_circle', 'gest_jerk_up', 'gest_jerk_down', 'gest_jerk_right', 'gest_jerk_left', 'gest_ccw_triangle', 'gest_cw_triangle', 'gest_zorro']

userid = raw_input('Enter user id: ')

# Read csv
print('Reading csv...')
for file_name in files:  # for each gesture
    x_accel, y_accel, z_accel, x_gyro, y_gyro, z_gyro, x_mag, y_mag, z_mag = ([] for i in range(9))
    with open(file_name) as gdata:
        data_csv = csv.reader(gdata, delimiter=',')
        for row in data_csv:
            sensors_data = map(float, row)
            x_accel.append(sensors_data[1])
            y_accel.append(sensors_data[2])
            z_accel.append(sensors_data[3])

            x_gyro.append(sensors_data[4])
            y_gyro.append(sensors_data[5])
            z_gyro.append(sensors_data[6])

            x_mag.append(sensors_data[10])
            y_mag.append(sensors_data[11])
            z_mag.append(sensors_data[12])

        xa_r = x_accel[::-1]
        ya_r = y_accel[::-1]
        za_r = z_accel[::-1]
        xg_r = x_gyro[::-1]
        yg_r = y_gyro[::-1]
        zg_r = z_gyro[::-1]
        xm_r = x_mag[::-1]
        ym_r = y_mag[::-1]
        zm_r = z_mag[::-1]
        print('Save csv file....')
        # file finished, seperate each data and save csv file
        for x in range(10):
            data = np.concatenate(np.atleast_2d(xa_r[x * 100: x * 100 + 100])[::-1],
                                  np.atleast_2d(ya_r[x * 100: x * 100 + 100])[::-1],
                                  np.atleast_2d(za_r[x * 100: x * 100 + 100])[::-1],
                                  np.atleast_2d(xg_r[x * 100: x * 100 + 100])[::-1],
                                  np.atleast_2d(yg_r[x * 100: x * 100 + 100])[::-1],
                                  np.atleast_2d(zg_r[x * 100: x * 100 + 100])[::-1],
                                  np.atleast_2d(xm_r[x * 100: x * 100 + 100])[::-1],
                                  np.atleast_2d(ym_r[x * 100: x * 100 + 100])[::-1],
                                  np.atleast_2d(zm_r[x * 100: x * 100 + 100])[::-1], axis=1)
            with open(gesture_list[files.index(file_name)] + '.csv', 'a') as gdata:
                np.savetxt(gdata, data, delimiter=',')
            gdata.close()
        plt.figure(1)
        plt.plot(z_accel)
        plt.figure(2)
        plt.plot(data)
        plt.show()
