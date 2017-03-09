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

# TODO: Random search for N_LSTM and N_dense

def train_and_record(N_LSTM, N_dense, b1, b2, lr):
    pass

if __name__ == '__main__':
    # Create list for hyperparameters
    N_LSTM_list = np.random.randint(10, 100, 20)
    N_dense_list = np.random.randint(5, 100, 20)
    b1_list = np.random.uniform(0.7, 1, 5)
    b2_list = np.random.uniform(0.7, 1, 5)
    lr_list = np.random.uniform(1e-5, 1e-3, 5)

    for N_LSTM in N_LSTM_list:
        for N_dense in N_dense_list:
            for b1 in b1_list:
                for b2 in b2_list:
                    for lr in lr_list:
    # Import the datas and seperate training vs test
    # split_data(0.2, 0.2)
    # y = load_data('labels')
    # split_index = int(np.ceil(test_ratio * len(y)))
    # user_ids = load_data('user_ids')

    # x_accel = load_data('x_acc_noise').reshape((y.shape[0], 100, 1))
    # y_accel = load_data('y_acc_noise').reshape((y.shape[0], 100, 1))
    # z_accel = load_data('z_acc_noise').reshape((y.shape[0], 100, 1))
    # x_gyro = load_data('x_gyr_noise').reshape((y.shape[0], 100, 1))
    # y_gyro = load_data('y_gyr_noise').reshape((y.shape[0], 100, 1))
    # z_gyro = load_data('z_gyr_noise').reshape((y.shape[0], 100, 1))

    # Split test and training data

    # x_accel_test = x_accel[split_index:, :]
    # print x_accel_test.shape
    # y_accel_test = y_accel[split_index:, :]
    # z_accel_test = z_accel[split_index:, :]
    # x_gyro_test = x_gyro[split_index:, :]
    # y_gyro_test = y_gyro[split_index:, :]
    # z_gyro_test = z_gyro[split_index:, :]
    # y_test = y[split_index:]
    # user_ids_test = user_ids[split_index:]

    # x_accel = x_accel[:split_index, :]
    # y_accel = y_accel[:split_index, :]
    # z_accel = z_accel[:split_index, :]
    # x_gyro = x_gyro[:split_index, :]
    # y_gyro = y_gyro[:split_index, :]
    # z_gyro = z_gyro[:split_index, :]
    # y = y[:split_index]
    # user_ids = user_ids[:split_index]

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
                        early_stopping = EarlyStopping(monitor='val_acc', patience=2)
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

                        # print 'Test Confusion Matrix'
                        # print confusion_matrix(y_test, predictions)
                        # print 'Test Accuracy: ', accuracy_score(y_test, predictions)
                        test_acc = accuracy_score(y_test, predictions)
                        # print '---'
                        # print 'Training Confusion Matrix'
                        # print confusion_matrix(y_train, predictions_train)
                        # print 'Train Accuracy: ', accuracy_score(y_train, predictions_train)
                        train_acc = accuracy_score(y_train, predictions_train)
                        val_acc = accuracy_score(y_validation, predictions_val)

                        # Record acccuracies of given values
                        acc_list = ['N_LSTM', N_LSTM, 'N_dense', N_dense, 'b1', b1, 'b2', b2, 'lr', lr, 'train_acc', train_acc, 'test_acc', test_acc, 'val_acc', val_acc]
                        acc_list2 = np.atleast_2d(np.asarray([N_LSTM, N_dense, b1, b2, lr, train_acc, test_acc, val_acc]))

                        with open('accuracies.csv', 'a') as gdata:
                            np.savetxt(gdata, acc_list2, delimiter=',', fmt='%d %d %3f %3f %3f %3f %3f %3f')
                        gdata.close()

    # print 'Trained Users: ', set(user_ids)
    # print 'Test Users: ', set(user_ids_test)
