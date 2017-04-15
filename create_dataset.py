import csv
import socket
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from collections import Counter
import pygame
import pygame.locals


if __name__ == '__main__':
    if len(sys.argv) < 2:
        ids = []
        with open('GestureDataset_Padded.csv', 'r') as gdata:
            data_csv = csv.reader(gdata, delimiter=',')
            for row in data_csv:
                ids.append((int(float(row[0])), int(float(row[1]))))
        gdata.close()
        for (gid, uid) in sorted(set(ids), key=ids.index):
            print 'User: ', uid, ', Gesture: ', gid, ', Collected: ', ids.count((gid, uid)), ', Left: ', 30 - ids.count((gid, uid))
        print 'Usage: python create_dataset.py gesture_id user_id'

    else:
        gesture_id = int(sys.argv[1])
        user_id = int(sys.argv[2])

        # 9 sevki altin - 26 r m
        # 8 nil akdede - 26 r f
        # 7 elif kina - 21 r f
        # 6 merve karagozoglu - 21 r f
        # 5 simin isleyici - 21 r f
        # 4 meltem isleyici - 54 r f
        # 3 kubra yildirim - 27 r f
        # 2 onur yildirim - 27 r m
        # 1 yalim isleyici - 26 (N/A) r

        # 1 ccw circle
        # 2 cw circle
        # 3 jerk up
        # 4 jerk down
        # 5 jerk right
        # 6 jerk left
        # 7 ccw triangle
        # 8 cw triangle
        # 9 zorro
        gest_list = ['na',
                     'ccw_circle',
                     'cw circle',
                     'jerk up',
                     'jerk down',
                     'jerk right',
                     'jerk left',
                     'ccw triangle',
                     'cw triangle',
                     'zorro']
        # pygame.init()
        # BLACK = (0,0,0)
        # WIDTH = 1280
        # HEIGHT = 1024
        # windowSurface = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)

        # windowSurface.fill(BLACK)

        ids = []
        with open('GestureDataset_Padded.csv', 'r') as gdata:
            data_csv = csv.reader(gdata, delimiter=',')
            for row in data_csv:
                ids.append((int(float(row[0])), int(float(row[1]))))
        # print Counter(ids)
        gdata.close()
        print 'User: ', user_id, ', Gesture: ', gesture_id, ', Collected: ', ids.count((gesture_id, user_id)), ',\033[93m Left: ', 30 - ids.count((gesture_id, user_id)), '\033[0m'
        print 'Current Gesture: ', gest_list[gesture_id]

        # TODO: Do not quit everytime. change user id and gesture id while running.
        high_b, high_a = butter(3, 0.01, 'highpass')

        x_accel = np.asarray([])
        y_accel = np.asarray([])
        z_accel = np.asarray([])

        x_gyro = np.asarray([])
        y_gyro = np.asarray([])
        z_gyro = np.asarray([])

        std_dev_x_accel = 0.00095
        std_dev_y_accel = 0.00110
        std_dev_z_accel = 0.00167
        std_dev_x_gyro = 0.00100
        std_dev_y_gyro = 0.00909
        std_dev_z_gyro = 0.00008

        sample_size = 100

        # while True:
        #     events = pygame.event.get()
        #     for event in events:
        #         if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
        #             # Start recording
        #         if event.type == pygame.KEYDOWN and event.key == pygame.K_a:
        #             print 'a'
        #             pygame.quit()
        #             sys.exit()

        # Recording by pressing enter
        # TODO: enter d to delete row number
        i = raw_input('Enter r to start recording:  ')
        print i
        if i is 'r':
            print 'Recording...'
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # UDP
            sock.bind(("", 10552))
            os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (0.2, 900))
            try:
                while True:
                    data_raw = sock.recv(1024)
                    data = [float(x.strip()) for x in data_raw.split(',')]
                    x_accel = np.append(x_accel, data[1])
                    y_accel = np.append(y_accel, data[2])
                    z_accel = np.append(z_accel, data[3])

                    x_gyro = np.append(x_gyro, data[4])
                    y_gyro = np.append(y_gyro, data[5])
                    z_gyro = np.append(z_gyro, data[6])

            except KeyboardInterrupt:
                os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (0.2, 250))
                # Filter the acceleration data to remove the gravity
                x_accel = lfilter(high_b, high_a, x_accel)
                y_accel = lfilter(high_b, high_a, y_accel)
                z_accel = lfilter(high_b, high_a, z_accel)

                x_accel = np.atleast_2d(x_accel)
                y_accel = np.atleast_2d(y_accel)
                z_accel = np.atleast_2d(z_accel)

                x_gyro = np.atleast_2d(x_gyro)
                y_gyro = np.atleast_2d(y_gyro)
                z_gyro = np.atleast_2d(z_gyro)
                # Fix the size of the data to 100 samples (i.e. 2 seconds of data)
                if x_accel.size < sample_size:
                    print 'Recieved ', x_accel.size, ' samples.'
                    x_accel_padded = np.append(np.random.normal(0, std_dev_x_accel, (sample_size - x_accel.size)), x_accel)
                    y_accel_padded = np.append(np.random.normal(0, std_dev_y_accel, (sample_size - x_accel.size)), y_accel)
                    z_accel_padded = np.append(np.random.normal(0, std_dev_z_accel, (sample_size - x_accel.size)), z_accel)

                    x_gyro_padded = np.append(np.random.normal(0, std_dev_x_gyro, (sample_size - x_accel.size)), x_gyro)
                    y_gyro_padded = np.append(np.random.normal(0, std_dev_y_gyro, (sample_size - x_accel.size)), y_gyro)
                    z_gyro_padded = np.append(np.random.normal(0, std_dev_z_gyro, (sample_size - x_accel.size)), z_gyro)
                    print 'Data is padded with noise to '+str(sample_size)+' samples.'

                    x_accel_padded = np.atleast_2d(x_accel_padded)
                    y_accel_padded = np.atleast_2d(y_accel_padded)
                    z_accel_padded = np.atleast_2d(z_accel_padded)

                    x_gyro_padded = np.atleast_2d(x_gyro_padded)
                    y_gyro_padded = np.atleast_2d(y_gyro_padded)
                    z_gyro_padded = np.atleast_2d(z_gyro_padded)

                    datas = np.concatenate((np.atleast_2d(np.asarray(gesture_id)), np.atleast_2d(np.asarray(user_id)),
                                            x_accel, y_accel, z_accel,
                                            x_gyro, y_gyro, z_gyro), axis=1)

                    datas_padded = np.concatenate((np.atleast_2d(np.asarray(gesture_id)), np.atleast_2d(np.asarray(user_id)),
                                                   x_accel_padded, y_accel_padded, z_accel_padded,
                                                   x_gyro_padded, y_gyro_padded, z_gyro_padded), axis=1)

                    print 'Saving csv file...'
                    with open('GestureDataset_Padded.csv', 'a') as gdata:
                        np.savetxt(gdata, datas_padded, delimiter=',')
                    gdata.close()

                    with open('GestureDataset.csv', 'a') as gdata:
                        np.savetxt(gdata, datas, delimiter=',')
                    gdata.close()
                    # np.savez('./gr_dataset_' + str(user_id) + '_' + str(gesture_id) + '.npz', x_accel, y_accel, z_accel, x_gyro, y_gyro, z_gyro)
                    print 'Saved!'

                else:
                    print 'Data too long please redo'
                    print datas.shape
        if i is 'p':
            with open('./gr_dataset.csv') as csvfile:
                sensors = csv.reader(csvfile, delimiter=',')
                for row in sensors:
                    sensors_data = map(float, row)
                    x_accel_plot = sensors_data[0:99]
                    y_accel_plot = sensors_data[100:199]
                    z_accel_plot = sensors_data[200:299]

                    plt.figure(1)
                    plt.plot(x_accel_plot)
                    plt.title('x_accel')

                    plt.figure(2)
                    plt.plot(y_accel_plot)
                    plt.title('y_accel')

                    plt.figure(3)
                    plt.plot(z_accel_plot)
                    plt.title('z_accel')
                    plt.show()
        # if i is 'd':
        #     file_selected = raw_input('Enter 1 for padded 2 for unpadded: ')
        #     rows = raw_input('Which row do you want to remove: ')

        #     if file_selected is '1':
        #         with open('GestureDataset_Padded', 'r') as fdata:
        #             print fdata

        #     if file_selected is '2':
        #         with open('GestureDataset', 'r') as fdata:
        #             print fdata
