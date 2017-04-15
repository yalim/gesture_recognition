import socket
import numpy as np
from scipy.signal import butter, lfilter
from utils import fifo_array
# from six.moves import cPickle
from keras.models import load_model
# from theano import shared


# class fifo_array():
#     """
#         A numpy ndarray that has a fixed length and works as FIFO
#     """
#     def __init__(self, max_length):
#         self.max_len = max_length
#         self.arr = np.zeros((1, self.max_len))

#     def add_element(self, element):
#         """
#             Adds one element to the end
#         """
#         self.arr2 = np.append(self.arr, element)
#         self.arr2 = np.delete(self.arr2, 0, 0)
#         self.arr = np.reshape(self.arr2, self.arr.shape)

#     def get_value(self):
#         return self.arr

#     def change_length(self, new_length):
#         """
#             This function adds zero to the end when increasing the length and
#         removes from beginning when decreasing length
#         """
#         if self.max_len <= new_length:
#             self.arr3 = np.zeros((1, new_length))
#             self.arr3[0, 0:self.max_len] = self.arr
#             self.arr = self.arr3
#             self.max_len = new_length
#         else:
#             x = [y for y in range(self.max_len - new_length)]
#             print x
#             self.arr4 = np.delete(self.arr, x, 1)
#             self.arr = self.arr4
#             self.max_len = new_length

if __name__ == '__main__':
    # Get data
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

    x_accel = fifo_array(100)
    y_accel = fifo_array(100)
    z_accel = fifo_array(100)
    x_gyro = fifo_array(100)
    y_gyro = fifo_array(100)
    z_gyro = fifo_array(100)
    high_b, high_a = butter(3, 0.01, 'highpass')

    # Load model
    lstm_gesture_classification = load_model("gr_keras_lstm_dense.h5")
    noise_vs_gesture = load_model('noise_vs_gesture_keras_model.h5')
    print 'Load models'

    while True:
        data_raw = sock.recv(1024)
        data = [float(x.strip()) for x in data_raw.split(',')]
        x_accel.add_element(data[1])
        y_accel.add_element(data[2])
        z_accel.add_element(data[3])
        x_gyro.add_element(data[4])
        y_gyro.add_element(data[5])
        z_gyro.add_element(data[6])

        # Filter the gravity term
        high_b, high_a = butter(3, 0.01, 'highpass')
        x_accel_np = lfilter(high_b, high_a, x_accel.get_value()).reshape((1, 100))
        y_accel_np = lfilter(high_b, high_a, y_accel.get_value()).reshape((1, 100))
        z_accel_np = lfilter(high_b, high_a, z_accel.get_value()).reshape((1, 100))

        # Predict noise vs gesture then gesture classification
        x = np.hstack((x_accel_np, y_accel_np, z_accel_np, x_gyro.get_value(), y_gyro.get_value(), z_gyro.get_value()))
        prediction = np.argmax(noise_vs_gesture.predict(x), axis=1)
        print prediction

        if prediction[0] is 1:
            print 'Gesture!!'
            # classification = np.argmax(lstm_gesture_classification.predict([x_accel_np.reshape((1, 100, 1)),
                                                                            # y_accel_np.reshape((1, 100, 1)),
                                                                            # z_accel_np.reshape((1, 100, 1)),
                                                                            # x_gyro.get_value().reshape((1, 100, 1)),
                                                                            # y_gyro.get_value().reshape((1, 100, 1)),
                                                                            # z_gyro.get_value().reshape((1, 100, 1))]), axis=1)

            print gest_list[classification]
        else:
            print 'Noise'
