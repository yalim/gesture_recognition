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
    subsample = 2
    N_LSTM_list = np.random.randint(10, 100, 10)
    N_dense_list = np.random.randint(5, 100, 10)
    b1_list = np.random.uniform(0.7, 1, 3)
    b2_list = np.random.uniform(0.7, 1, 3)
    lr_list = np.random.uniform(1e-5, 1e-3, 3)

    for N_LSTM in N_LSTM_list:
        for N_dense in N_dense_list:
            for b1 in b1_list:
                for b2 in b2_list:
                    for lr in lr_list:
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

                        with open('accuracies_25hz.csv', 'a') as gdata:
                            np.savetxt(gdata, acc_list2, delimiter=',', fmt='%d %d %3f %3f %3f %3f %3f %3f')
                        gdata.close()

                        # final_model.save('gr_keras_lstm_dense_'+str(N_LSTM)+'_'+str(N_dense)+'_'+str(b1)+'_'+str(b2)+'_'+str(lr)+'h5')


    # print 'Trained Users: ', set(user_ids)
    # print 'Test Users: ', set(user_ids_test)
