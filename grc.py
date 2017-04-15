from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.optimizers import Nadam
from keras.layers import Merge
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.utils.np_utils import to_categorical
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from import_dataset import x_accel_test, x_accel_validation, x_accel_train, y_accel_test, y_accel_validation, y_accel_train, z_accel_test, z_accel_validation, z_accel_train, x_gyro_test, x_gyro_validation, x_gyro_train, y_gyro_test, y_gyro_validation, y_gyro_train, z_gyro_test, z_gyro_validation, z_gyro_train, y_test, y_validation, y_train

if __name__ == '__main__':
    N_dense = 99
    N_LSTM = 44
    lr = 9.7e-4
    b1 = 0.841
    b2 = 0.887
    model_x_acc = Sequential()
    model_x_acc.add(LSTM(N_LSTM, input_shape=(100, 1)))
    model_y_acc = Sequential()
    model_y_acc.add(LSTM(N_LSTM, input_shape=(100, 1)))
    model_z_acc = Sequential()
    model_z_acc.add(LSTM(N_LSTM, input_shape=(100, 1)))

    model_x_gyr = Sequential()
    model_x_gyr.add(LSTM(N_LSTM, input_shape=(100, 1)))
    model_y_gyr = Sequential()
    model_y_gyr.add(LSTM(N_LSTM, input_shape=(100, 1)))
    model_z_gyr = Sequential()
    model_z_gyr.add(LSTM(N_LSTM, input_shape=(100, 1)))

    merged = Merge([model_x_acc,
                    model_y_acc,
                    model_z_acc,
                    model_x_gyr,
                    model_y_gyr,
                    model_z_gyr],
                   mode='concat')

    final_model = Sequential()
    final_model.add(merged)
    final_model.add(Dense(N_dense, activation='relu'))
    final_model.add(Dense(9, activation='softmax'))
    early_stopping = EarlyStopping(monitor='val_acc', patience=1)
    nadam = Nadam(lr=lr, beta_1=b1, beta_2=b2)
    final_model.compile(optimizer=nadam, loss='categorical_crossentropy', metrics=['accuracy'])
    y_train_c = to_categorical(y_train)
    y_validation_c = to_categorical(y_validation)

    final_model.fit([x_accel_train,
                     y_accel_train,
                     z_accel_train,
                     x_gyro_train,
                     y_gyro_train,
                     z_gyro_train],
                    y_train_c,
                    validation_data=([x_accel_validation,
                                     y_accel_validation,
                                     z_accel_validation,
                                     x_gyro_validation,
                                     y_gyro_validation,
                                     z_gyro_validation],
                                     y_validation_c),
                    nb_epoch=1000,
                    callbacks=[early_stopping])

    predictions = np.argmax(final_model.predict([x_accel_test,
                                                 y_accel_test,
                                                 z_accel_test,
                                                 x_gyro_test,
                                                 y_gyro_test,
                                                 z_gyro_test]), axis=1)

    predictions_train = np.argmax(final_model.predict([x_accel_train,
                                                       y_accel_train,
                                                       z_accel_train,
                                                       x_gyro_train,
                                                       y_gyro_train,
                                                       z_gyro_train]), axis=1)

    predictions_val = np.argmax(final_model.predict([x_accel_validation,
                                                     y_accel_validation,
                                                     z_accel_validation,
                                                     x_gyro_validation,
                                                     y_gyro_validation,
                                                     z_gyro_validation]), axis=1)

    print 'Test Confusion Matrix'
    print confusion_matrix(y_test, predictions)
    print 'Test Accuracy: ', accuracy_score(y_test, predictions)
    test_acc = accuracy_score(y_test, predictions)
    print '---'
    print 'Training Confusion Matrix'
    print confusion_matrix(y_train, predictions_train)
    print 'Train Accuracy: ', accuracy_score(y_train, predictions_train)
    train_acc = accuracy_score(y_train, predictions_train)
    val_acc = accuracy_score(y_validation, predictions_val)
    final_model.save("gr_keras_lstm_dense.h5")
