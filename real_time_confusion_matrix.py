"""Realtime gesture recognition and real time confusion matrix."""
import socket
import numpy as np
from scipy.signal import butter, lfilter
from keras.models import load_model
from utils import fifo_array
from utils import bcolors
from keras.models import Sequential
from keras.layers import Merge
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
# import warnings


def most_common(lst):
    """Find the most common element in a list."""
    return max(set(lst), key=lst.count)

if __name__ == '__main__':
    # Initialize the final model and load weights
    # warnings.simfplefilter('default')
    subsample = 1
    N_dense = 99  # 60 for subsample 2
    N_LSTM = 44  # 40 for subsample 2
    model_x_acc = Sequential()
    model_x_acc.add(LSTM(N_LSTM, input_shape=(100 / subsample, 1)))
    model_y_acc = Sequential()
    model_y_acc.add(LSTM(N_LSTM, input_shape=(100 / subsample, 1)))
    model_z_acc = Sequential()
    model_z_acc.add(LSTM(N_LSTM, input_shape=(100 / subsample, 1)))

    model_x_gyr = Sequential()
    model_x_gyr.add(LSTM(N_LSTM, input_shape=(100 / subsample, 1)))
    model_y_gyr = Sequential()
    model_y_gyr.add(LSTM(N_LSTM, input_shape=(100 / subsample, 1)))
    model_z_gyr = Sequential()
    model_z_gyr.add(LSTM(N_LSTM, input_shape=(100 / subsample, 1)))

    merged = Merge([model_x_acc,
                    model_y_acc,
                    model_z_acc,
                    model_x_gyr,
                    model_y_gyr,
                    model_z_gyr],
                   mode='concat')

    lstm_gesture_classification = Sequential()
    lstm_gesture_classification.add(merged)
    lstm_gesture_classification.add(Dense(N_dense, activation='relu'))
    lstm_gesture_classification.add(Dense(9, activation='softmax'))

    lstm_gesture_classification.load_weights('gr_keras_weights.h5')

    print('Model initialized, weights loaded.')

    noise_vs_gesture = load_model('noise_vs_gesture_keras_model.h5')
    print('Gesture vs Noise model loaded')

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # UDP
    sock.bind(("", 10552))

    # Definitions
    gest_list = ['ccw_circle',
                 'cw circle',
                 'jerk up',
                 'jerk down',
                 'jerk right',
                 'jerk left',
                 'ccw triangle',
                 'cw triangle',
                 'zorro']

    x_accel = fifo_array(120)
    y_accel = fifo_array(120)
    z_accel = fifo_array(120)
    x_gyro = fifo_array(120)
    y_gyro = fifo_array(120)
    z_gyro = fifo_array(120)
    gesture_prediction_list = fifo_array(100)
    high_b, high_a = butter(3, 0.01, 'highpass')

    energy_final_prev = 0
    conf_matrix = np.zeros((9, 9))
    gesture_vote = []

    for gest_id in range(9):  # Select gesture for filling the confusion matrix.
        print(bcolors.WARNING + 'Do gesture: ' + gest_list[gest_id] + bcolors.ENDC)
        # print('Gest ID: ', gest_id)
        gest_count = 0
        while True:
            data_raw = sock.recv(1024)  # Recieve data
            data = [float(x.strip()) for x in data_raw.split(',')]
            x_accel.add_element(data[1])
            y_accel.add_element(data[2])
            z_accel.add_element(data[3])
            x_gyro.add_element(data[4])
            y_gyro.add_element(data[5])
            z_gyro.add_element(data[6])

            # Filter the gravity term and convert fifos to numpy arrays
            high_b, high_a = butter(3, 0.01, 'highpass')
            x_accel_np = lfilter(high_b, high_a, x_accel.get_value())[0, 0:100:subsample].reshape((1, 100 / subsample))
            y_accel_np = lfilter(high_b, high_a, y_accel.get_value())[0, 0:100:subsample].reshape((1, 100 / subsample))
            z_accel_np = lfilter(high_b, high_a, z_accel.get_value())[0, 0:100:subsample].reshape((1, 100 / subsample))
            x_gyro_np = x_gyro.get_value()[0, 0:100:subsample].reshape((1, 100 / subsample))
            y_gyro_np = y_gyro.get_value()[0, 0:100:subsample].reshape((1, 100 / subsample))
            z_gyro_np = z_gyro.get_value()[0, 0:100:subsample].reshape((1, 100 / subsample))

            # Predict noise vs gesture then gesture classification

            x = np.hstack((x_accel_np, y_accel_np, z_accel_np, x_gyro_np, y_gyro_np, z_gyro_np))
            x_60 = np.hstack((x_gyro.get_value()[0, 100:], x_gyro.get_value()[0, 100:], x_gyro.get_value()[0, 100:]))
            # print(x_60)
            std_dev = np.std(x_60)
            # print(std_dev)

            prediction = np.argmax(noise_vs_gesture.predict(x), axis=1)
            probability_of_gesture = noise_vs_gesture.predict(x)
            # print('Noise vs gesture prob: ' + str(probability_of_gesture))

            if probability_of_gesture[0, 1] > 0.9:  # Put a threshold to gesture probability
                gesture_detected = True
                # print(bcolors.WARNING + 'Gesture!!' + bcolors.ENDC)

                probabilities = lstm_gesture_classification.predict([x_accel_np.reshape((1, 100 / subsample, 1)),
                                                    y_accel_np.reshape((1, 100 / subsample, 1)),
                                                    z_accel_np.reshape((1, 100 / subsample, 1)),
                                                    x_gyro_np.reshape((1, 100 / subsample, 1)),
                                                    y_gyro_np.reshape((1, 100 / subsample, 1)),
                                                    z_gyro_np.reshape((1, 100 / subsample, 1))])

                if np.amax(probabilities, axis=1) > 0.8 and std_dev < 0.5:  # Check sureness of the gesture detection and if gesture is finished.
                    classification = np.argmax(probabilities, axis=1)
                    gesture_vote.append(classification[0])
                    # print(bcolors.WARNING + 'Probability: ' +
                    #       str(np.amax(probabilities, axis=1)) +
                    #       'Gesture: ' + str(gest_list[classification[0]]) + bcolors.ENDC)
            elif gesture_vote and std_dev < 1.0:
                # print(gesture_vote)
                selected_gesture = most_common(gesture_vote)
                gest_count += 1
                print(gest_count)
                print('Selected Gesture: ', gest_list[selected_gesture])
                conf_matrix[gest_id, selected_gesture] = conf_matrix[gest_id, selected_gesture] + 1
                gesture_vote = []

            if gest_count is 10:
                break
        print(conf_matrix)
    print('Accuracy: ' + str(np.trace(conf_matrix) / np.sum(conf_matrix)))

# 50 Hz
# [[ 7.  1.  1.  0.  0.  0.  0.  1.  0.]
#  [ 1.  6.  2.  0.  0.  0.  0.  1.  0.]
#  [ 4.  3.  0.  0.  0.  0.  0.  3.  0.]
#  [ 0.  1.  6.  1.  0.  0.  0.  0.  2.]
#  [ 0.  4.  0.  0.  0.  0.  0.  6.  0.]
#  [ 2.  0.  0.  0.  1.  5.  0.  2.  0.]
#  [ 5.  0.  0.  0.  0.  0.  0.  5.  0.]
#  [ 0.  1.  0.  0.  0.  0.  0.  9.  0.]
#  [ 0.  2.  0.  0.  0.  1.  0.  6.  1.]]
# 0.32

# [[ 5.  0.  3.  0.  0.  0.  2.  0.  0.]
#  [ 0.  2.  1.  0.  0.  1.  0.  6.  0.]
#  [ 2.  3.  1.  0.  0.  0.  0.  4.  0.]
#  [ 0.  3.  5.  2.  0.  0.  0.  0.  0.]
#  [ 0.  2.  0.  1.  3.  0.  0.  4.  0.]
#  [ 0.  1.  0.  0.  1.  3.  0.  2.  3.]
#  [ 0.  0.  0.  0.  0.  0.  5.  4.  1.]
#  [ 1.  0.  1.  0.  0.  0.  3.  5.  0.]
#  [ 0.  2.  0.  0.  0.  0.  0.  7.  1.]]
# Accuracy: 0.3

# CEMRE
# [[ 10.   0.   0.   0.   0.   0.   0.   0.   0.]
#  [  1.   9.   0.   0.   0.   0.   0.   0.   0.]
#  [  1.   0.   6.   2.   0.   0.   0.   1.   0.]
#  [  0.   0.   2.   4.   0.   0.   0.   4.   0.]
#  [  0.   0.   0.   0.   7.   0.   0.   2.   1.]
#  [  0.   0.   1.   0.   0.   7.   0.   2.   0.]
#  [  1.   0.   0.   0.   0.   0.   4.   5.   0.]
#  [  1.   0.   0.   0.   0.   0.   0.   9.   0.]
#  [  1.   0.   0.   0.   0.   0.   4.   2.   3.]]
# Accuracy: 0.655555555556


# 25 Hz
# [[  2.   0.   0.   0.   0.   0.   1.   7.   0.]
#  [  0.   6.   0.   0.   0.   0.   0.   4.   0.]
#  [  0.   7.   0.   0.   0.   0.   0.   3.   0.]
#  [  0.   0.   0.   0.   0.   0.   2.   8.   0.]
#  [  0.   0.   0.   0.   0.   0.   0.  10.   0.]
#  [  0.   0.   0.   0.   0.   0.   8.   2.   0.]
#  [  2.   0.   0.   0.   0.   0.   1.   7.   0.]
#  [  0.   4.   0.   0.   0.   0.   0.   6.   0.]
#  [  0.   0.   0.   0.   0.   0.   1.   9.   0.]]
# 0.16
