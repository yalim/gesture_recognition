import socket
import numpy as np
from scipy.signal import butter, lfilter
from utils import fifo_array
from six.moves import cPickle
from keras.models import load_model
from theano import shared


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

    # Load gesture vs noise model
    f = open('ann_gr.save', 'rb')
    gesture_vs_noise = cPickle.load(f)
    f.close()
    print 'Gesture vs Noise loaded'

    # Load LSTM for gesture classification
    lstm_gesture_classification = load_model("gr_keras_lstm_dense.h5")

    while True:
        data_raw = sock.recv(1024)
        data = [float(x.strip()) for x in data_raw.split(',')]
        x_accel = x_accel.add_element(data[1])
        y_accel = y_accel.add_element(data[2])
        z_accel = z_accel.add_element(data[3])
        x_gyro = x_gyro.add_element(data[4])
        y_gyro = y_gyro.add_element(data[5])
        z_gyro = z_gyro.add_element(data[6])

    # Filter the gravity term
    high_b, high_a = butter(3, 0.01, 'highpass')
    x_accel = lfilter(high_b, high_a, x_accel)
    y_accel = lfilter(high_b, high_a, y_accel)
    z_accel = lfilter(high_b, high_a, z_accel)

    # Predict noise vs gesture then gesture classification
    x = shared(np.hstack(x_accel, y_accel, z_accel, x_gyro, y_gyro, z_gyro))
    prediction = gesture_vs_noise.preditct(x)

    if prediction is 1:
        classification = np.argmax(lstm_gesture_classification.preditct([x_accel,
                                                                         y_accel,
                                                                         z_accel,
                                                                         x_gyro,
                                                                         y_gyro,
                                                                         z_gyro]), axis=1)

        print gest_list(classification)
    else:
        print 'Noise'
