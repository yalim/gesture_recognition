import csv
import socket
import numpy as np
import sys
import os

if __name__ == '__main__':

    # Default values
    sample_size = 100
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

    try:
        ids =[]
        gest_list = ['ccw_circle',
                     'cw circle',
                     'jerk up',
                     'jerk down',
                     'jerk right',
                     'jerk left',
                     'ccw triangle',
                     'cw triangle',
                     'zorro']
        with open('GestureDataset_Padded.csv', 'r') as gdata:
            data_csv = csv.reader(gdata, delimiter=',')
            for row in data_csv:
                ids.append((int(float(row[0])), int(float(row[1]))))
            gdata.close()
        for (gid, uid) in sorted(set(ids), key=ids.index):
            print 'User: ', uid, ', Gesture: ', gid, ', Collected: ', ids.count((gid, uid)), ', Left: ', 30 - ids.count((gid, uid))
    
        while True:
            user_id = int(raw_input('Enter user id or Ctrl + C to exit: '))
            try:
                while True:
                    gesture_id = int(raw_input('Enter gesture id or Ctrl + C to exit: '))
                    print 'Selected gesture: ', gest_list[gesture_id]
                    print 'User: ', user_id, ', Gesture: ', gesture_id, ', Collected: ', ids.count((gesture_id, user_id)), ',\033[93m Left: ', 30 - ids.count((gesture_id, user_id)), '\033[0m'
                    raw_input('Continue to record? (Press return to continue or Ctrl + C to exit.')
                    try:
                        # Record the values
                        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                        sock.bind(("", 10552))
                        print 'Recording...'
                        os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (0.2, 900))
                        while True:
                            data_raw = sock.recv(1024)
                            data = [float(x.strip()) for x in data.raw.split(',')]
                            x_accel = np.append(x_accel, data[1])
                            y_accel = np.append(y_accel, data[2])
                            z_accel = np.append(z_accel, data[3])

                            x_gyro = np.append(x_gyro, data[4])
                            y_gyro = np.append(y_gyro, data[5])
                            z_gyro = np.append(z_gyro, data[6])

                    except KeyboardInterrupt:
                        os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (0.2, 250))
                        x_accel = np.atleast_2d(x_accel)
                        y_accel = np.atleast_2d(y_accel)
                        z_accel = np.atleast_2d(z_accel)

                        x_gyro = np.atleast_2d(x_gyro)
                        y_gyro = np.atleast_2d(y_gyro)
                        z_gyro = np.atleast_2d(z_gyro)
                        # Save the dataset

                        if x_accel.size < sample_size:
                            print '\nRecieved ', x_accel.size, ' samples.'
                            x_accel_padded = np.append(np.random.normal(0, std_dev_x_accel, (sample_size - x_accel.size)), x_accel)
                            y_accel_padded = np.append(np.random.normal(0, std_dev_y_accel, (sample_size - x_accel.size)), y_accel)
                            z_accel_padded = np.append(np.random.normal(0, std_dev_z_accel, (sample_size - x_accel.size)), z_accel)

                            x_gyro_padded = np.append(np.random.normal(0, std_dev_x_gyro, (sample_size - x_accel.size)), x_gyro)
                            y_gyro_padded = np.append(np.random.normal(0, std_dev_y_gyro, (sample_size - x_accel.size)), y_gyro)
                            z_gyro_padded = np.append(np.random.normal(0, std_dev_z_gyro, (sample_size - x_accel.size)), z_gyro)
                            print 'Data is padded with noise to '+str(x_accel_padded.shape)+' samples.'

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
                            with open('GestureDataset_Padded_2.csv', 'a') as gdata:
                                np.savetxt(gdata, datas_padded, delimiter=',')
                            gdata.close()

                            with open('GestureDataset_2.csv', 'a') as gdata:
                                np.savetxt(gdata, datas, delimiter=',')
                            gdata.close()
                            print 'Saved!'
                        else:
                            print 'Data too long!!'

            except KeyboardInterrupt:
                pass

    except KeyboardInterrupt:
        pass
