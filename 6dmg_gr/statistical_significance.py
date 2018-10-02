import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
from janet import JANET
from keras.layers.convolutional import Conv1D
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Nadam
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import TimeDistributed, GlobalMaxPooling1D
from keras.utils.np_utils import to_categorical
from keras.utils import plot_model
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from keras.layers.recurrent import LSTM
from keras.preprocessing.sequence import pad_sequences
from scipy.io import loadmat
import os
import sqlite3
# from grc2 import plot_confusion_matrix


# np.set_printoptions(precision=3, threshold=np.nan, suppress=True)
# yTrain = []
# xVal = []
# yVal = []
# xTest = []
# yTest = []
# xAcc = []
# yAcc = []
# zAcc = []
# xGyr = []
# yGyr = []
# zGyr = []

# # Training set
# file_loc = ['../6DMG/matR/', '../6DMG/matL/']
# gest_list = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19']
# gest_categorical = to_categorical(gest_list)
# # user_list = ['J3', 'Y3', 'F1', 'D2', 'M2', 'S2', 'T2', 'B2', 'J4', 'R2', 'J2', 'M1', 'C1', 'Y2', 'M3', 'S3', 'W2', 'D1', 'S1', 'U1', 'W1', 'T1', 'B1', 'J5', 'R1', 'J1', 'Y1', 'C2']
# user_list = ['M3', 'S3', 'W2', 'D1', 'S1', 'U1', 'W1', 'T1', 'B1', 'J5', 'R1', 'J1', 'Y1', 'C2']
# repetition_list = ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10']
# for trial in repetition_list:
#     for gest, location in zip(gest_list, range(len(gest_list))):
#         for tester in user_list:
#             for loc in file_loc:
#                 file_name = loc + 'g' + gest +'_'+ tester + '_' + trial + '.mat'
#                 if os.path.isfile(file_name):
#                     data = loadmat(file_name)['gest']
#                     xAcc.append(data[8, :].T)
#                     yAcc.append(data[9, :].T)
#                     zAcc.append(data[10, :].T)

#                     xGyr.append(data[11, :].T)
#                     yGyr.append(data[12, :].T)
#                     zGyr.append(data[13, :].T)
#                     yTrain.append(gest_categorical[location])

# xAccPadded = pad_sequences(xAcc, maxlen=240)
# yAccPadded = pad_sequences(yAcc, maxlen=240)
# zAccPadded = pad_sequences(zAcc, maxlen=240)

# xGyrPadded = pad_sequences(xGyr, maxlen=240)
# yGyrPadded = pad_sequences(yGyr, maxlen=240)
# zGyrPadded = pad_sequences(zGyr, maxlen=240)

# xTrain = np.dstack((xAccPadded, yAccPadded, zAccPadded, xGyrPadded, yGyrPadded, zGyrPadded))
# yTrain = np.asarray(yTrain)
# print xTrain.shape

# # Validation set
# user_list = ['R2', 'J2', 'M1', 'C1', 'Y2']

# xAcc = []
# yAcc = []
# zAcc = []
# xGyr = []
# yGyr = []
# zGyr = []

# for trial in repetition_list:
#     for gest, location in zip(gest_list, range(len(gest_list))):
#         for tester in user_list:
#             for loc in file_loc:
#                 file_name = loc + 'g' + gest +'_'+ tester + '_' + trial + '.mat'
#                 if os.path.isfile(file_name):
#                     data = loadmat(file_name)['gest']
#                     xAcc.append(data[8, :].T)
#                     yAcc.append(data[9, :].T)
#                     zAcc.append(data[10, :].T)

#                     xGyr.append(data[11, :].T)
#                     yGyr.append(data[12, :].T)
#                     zGyr.append(data[13, :].T)
#                     yVal.append(gest_categorical[location])

# xAccPadded = pad_sequences(xAcc, maxlen=240)
# yAccPadded = pad_sequences(yAcc, maxlen=240)
# zAccPadded = pad_sequences(zAcc, maxlen=240)

# xGyrPadded = pad_sequences(xGyr, maxlen=240)
# yGyrPadded = pad_sequences(yGyr, maxlen=240)
# zGyrPadded = pad_sequences(zGyr, maxlen=240)

# xVal = np.dstack((xAccPadded, yAccPadded, zAccPadded, xGyrPadded, yGyrPadded, zGyrPadded))
# print xVal.shape
# yVal = np.asarray(yVal)

# # Test set
# user_list = ['J3', 'Y3', 'F1', 'D2', 'M2', 'S2', 'T2', 'B2', 'J4']

# xAcc = []
# yAcc = []
# zAcc = []
# xGyr = []
# yGyr = []
# zGyr = []

# for trial in repetition_list:
#     for gest, location in zip(gest_list, range(len(gest_list))):
#         for tester in user_list:
#             for loc in file_loc:
#                 file_name = loc + 'g' + gest +'_'+ tester + '_' + trial + '.mat'
#                 if os.path.isfile(file_name):
#                     data = loadmat(file_name)['gest']
#                     xAcc.append(data[8, :].T)
#                     yAcc.append(data[9, :].T)
#                     zAcc.append(data[10, :].T)

#                     xGyr.append(data[11, :].T)
#                     yGyr.append(data[12, :].T)
#                     zGyr.append(data[13, :].T)
#                     yTest.append(np.argmax(np.atleast_2d(gest_categorical[location])))

# xAccPadded = pad_sequences(xAcc, maxlen=240)
# yAccPadded = pad_sequences(yAcc, maxlen=240)
# zAccPadded = pad_sequences(zAcc, maxlen=240)

# xGyrPadded = pad_sequences(xGyr, maxlen=240)
# yGyrPadded = pad_sequences(yGyr, maxlen=240)
# zGyrPadded = pad_sequences(zGyr, maxlen=240)

# xTest = np.dstack((xAccPadded, yAccPadded, zAccPadded, xGyrPadded, yGyrPadded, zGyrPadded))
# # print xTest.shape
# # print type(xTrain)
# print yTrain.shape

# np.savez('GestureDatasetPadded.npz', xTrain=xTrain, yTrain=yTrain, xVal=xVal, yVal=yVal, xTest=xTest, yTest=yTest)

# Begin statistical significance tests

npzfile = np.load('GestureDatasetPadded.npz')
xTrain = npzfile['xTrain']
yTrain = npzfile['yTrain']
xVal = npzfile['xVal']
yVal = npzfile['yVal']
xTest = npzfile['xTest']
yTest = npzfile['yTest']


repetition = 50
epochs = 100
patience = 3
batch_size=100

# Model 1
C = [78, 54, 76, 52, 27, 52, 35, 36, 37, 75]
J = [77, 31, 61, 23, 74, 16, 49, 55, 49, 17]
D = [27, 72, 70, 74, 57, 72, 19, 30, 75, 57]

for NConv, NJanet, NDense in zip(C, J, D):
    acc_list = []
    prec_list = []
    recall_list = []
    f1_list = []
    support_list = []
    for i in range(repetition):
        early_stopping = EarlyStopping(monitor='val_acc', patience=patience)
        checkpoint = ModelCheckpoint('6dmg_try.h5', monitor='val_acc', verbose=0, save_weights_only=True)
        nadam = Nadam(lr=1e-4, beta_1=0.9, beta_2=0.9)

        model1 = Sequential()
        model1.add(Conv1D(NConv, 5, activation='relu', input_shape=(240, 6)))
        model1.add(JANET(NJanet, activation='relu')) #, input_shape=(None, 6)))
        model1.add(Dense(NDense, activation='relu'))
        model1.add(Dense(20, activation='softmax'))
        print model1.summary(90)
        model1.compile(loss='categorical_crossentropy', optimizer=nadam, metrics=['accuracy'])
        model1.fit(xTrain,
                   yTrain,
                   epochs=epochs,
                   validation_data=(xVal, yVal),
                   callbacks=[early_stopping],
                   verbose=1,
                   batch_size=batch_size)

        predictedTest = np.argmax(model1.predict(xTest), axis=1)
        acc = accuracy_score(yTest, predictedTest)
        precision, recall, f1, support = precision_recall_fscore_support(yTest, predictedTest, average='weighted')
        acc_list.append(acc)
        prec_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    acc_mean = np.mean(np.asarray(acc_list))
    prec_mean = np.mean(np.asarray(prec_list))
    recall_mean = np.mean(np.asarray(recall_list))
    f1_mean = np.mean(np.asarray(f1_list))

    acc_std = np.std(np.asarray(acc_list))
    prec_std = np.std(np.asarray(prec_list))
    recall_std = np.std(np.asarray(recall_list))
    f1_std = np.std(np.asarray(f1_list))

    my_file = file('stat_CONV_JANET_DENSE.csv', 'a')
    params = np.asarray([NConv, NJanet, NDense, acc_mean, acc_std, prec_mean, prec_std, recall_mean, recall_std, f1_mean, f1_std])
    np.savetxt(my_file, params.reshape((1, len(params))), fmt='%.2f')
    my_file.close()

# Model 2
C1 = [22, 53, 28, 27, 66, 50, 49, 43, 24, 60]
C2 = [73, 72, 75, 65, 79, 59, 47, 77, 56, 66]
J = [55, 79, 72, 64, 69, 75, 61, 40, 76, 62]
D = [56, 76, 70, 30, 22, 79, 46, 56, 23, 73]

for NConv1, NConv2, NJanet, NDense in zip(C1, C2, J, D):
    acc_list = []
    prec_list = []
    recall_list = []
    f1_list = []
    for i in range(repetition):
        early_stopping = EarlyStopping(monitor='val_acc', patience=patience)
        checkpoint = ModelCheckpoint('6dmg_try.h5', monitor='val_acc', verbose=0, save_weights_only=True)
        nadam = Nadam(lr=1e-4, beta_1=0.9, beta_2=0.9)
        model2 = Sequential()
        model2.add(Conv1D(NConv1, 5, activation='relu', input_shape=(240, 6)))
        model2.add(Conv1D(NConv2, 5, activation='relu'))
        model2.add(JANET(NJanet, activation='relu'))
        model2.add(Dense(NDense, activation='relu'))
        model2.add(Dense(20, activation='softmax'))
        model2.compile(loss='categorical_crossentropy', optimizer=nadam, metrics=['accuracy'])
        model2.fit(xTrain,
                   yTrain,
                   epochs=epochs,
                   validation_data=(xVal, yVal),
                   callbacks=[early_stopping],
                   verbose=1,
                   batch_size=batch_size)

        predictedTest = np.argmax(model2.predict(xTest), axis=1)
        acc = accuracy_score(yTest, predictedTest)
        precision, recall, f1, support = precision_recall_fscore_support(yTest, predictedTest, average='weighted')
        acc_list.append(acc)
        prec_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    acc_mean = np.mean(np.asarray(acc_list))
    prec_mean = np.mean(np.asarray(prec_list))
    recall_mean = np.mean(np.asarray(recall_list))
    f1_mean = np.mean(np.asarray(f1_list))

    acc_std = np.std(np.asarray(acc_list))
    prec_std = np.std(np.asarray(prec_list))
    recall_std = np.std(np.asarray(recall_list))
    f1_std = np.std(np.asarray(f1_list))

    my_file = file('stat_CONV_CONV_JANET_DENSE.csv', 'a')
    params = np.asarray([NConv1, NConv2, NJanet, NDense, acc_mean, acc_std, prec_mean, prec_std, recall_mean, recall_std, f1_mean, f1_std])
    np.savetxt(my_file, params.reshape((1, len(params))), fmt='%.2f')
    my_file.close()

# Model 3
# C1 = [71, 28, 78, 67, 60, 28, 11, 75, 29, 71]
# C2 = [66, 68, 33, 59, 75, 58, 46, 51, 63, 78]
# J1 = [43, 60, 77, 68, 42, 15, 39, 66, 43, 26]
# J2 = [30, 27, 54, 41, 28, 25, 34, 42, 46, 50]
# D = [44, 40, 45, 70, 20, 67, 36, 43, 65, 63]
# for NConv1, NConv2, NJanet1, NJanet2, NDense in zip(C1, C2, J1, J2, D):
#     acc_list = []
#     prec_list = []
#     recall_list = []
#     f1_list = []
#     for i in range(repetition):
#         early_stopping = EarlyStopping(monitor='val_acc', patience=patience)
#         checkpoint = ModelCheckpoint('6dmg_try.h5', monitor='val_acc', verbose=0, save_weights_only=True)
#         nadam = Nadam(lr=1e-4, beta_1=0.9, beta_2=0.9)

#         model3 = Sequential()
#         model3.add(Conv1D(NConv1, 5, activation='relu', input_shape=(240, 6)))
#         model3.add(Conv1D(NConv2, 5, activation='relu'))
#         model3.add(JANET(NJanet1, activation='relu', return_sequences=True)) #, input_shape=(None, 6)))
#         model3.add(JANET(NJanet2, activation='relu'))
#         model3.add(Dense(NDense, activation='relu'))
#         model3.add(Dense(20, activation='softmax'))
#         model3.compile(loss='categorical_crossentropy', optimizer=nadam, metrics=['accuracy'])
#         model3.fit(xTrain,
#                    yTrain,
#                    epochs=epochs,
#                    validation_data=(xVal, yVal),
#                    callbacks=[early_stopping],
#                    verbose=1,
#                    batch_size=batch_size)

#         predictedTest = np.argmax(model3.predict(xTest), axis=1)
#         acc = accuracy_score(yTest, predictedTest)
#         precision, recall, f1, support = precision_recall_fscore_support(yTest, predictedTest, average='weighted')
#         acc_list.append(acc)
#         prec_list.append(precision)
#         recall_list.append(recall)
#         f1_list.append(f1)

#     acc_mean = np.mean(np.asarray(acc_list))
#     prec_mean = np.mean(np.asarray(prec_list))
#     recall_mean = np.mean(np.asarray(recall_list))
#     f1_mean = np.mean(np.asarray(f1_list))

#     acc_std = np.std(np.asarray(acc_list))
#     prec_std = np.std(np.asarray(prec_list))
#     recall_std = np.std(np.asarray(recall_list))
#     f1_std = np.std(np.asarray(f1_list))

#     my_file = file('stat_CONV_CONV_JANET_JANET_DENSE.csv', 'a')
#     params = np.asarray([NConv1, NConv2, NJanet1, NJanet2, NDense, acc_mean, acc_std, prec_mean, prec_std, recall_mean, recall_std, f1_mean, f1_std])
#     np.savetxt(my_file, params.reshape((1, len(params))), fmt='%.2f')
#     my_file.close()

# # Model 4
# C1 = [59, 71, 30, 43, 76, 38, 39, 53, 79, 69]
# C2 = [79, 66, 58, 50, 71, 22, 29, 48, 43, 57]
# D1 = [52, 18, 46, 37, 23, 48, 31, 77, 48, 73]
# D2 = [60, 51, 65, 53, 19, 48, 35, 50, 40, 37]
# for NConv1, NConv2, NDense1, NDense2 in zip(C1, C2, D1, D2):
#     acc_list = []
#     prec_list = []
#     recall_list = []
#     f1_list = []
#     for i in range(repetition):
#         early_stopping = EarlyStopping(monitor='val_acc', patience=patience)
#         checkpoint = ModelCheckpoint('6dmg_try.h5', monitor='val_acc', verbose=0, save_weights_only=True)
#         nadam = Nadam(lr=1e-4, beta_1=0.9, beta_2=0.9)

#         model4 = Sequential()
#         model4.add(Conv1D(NConv1, 5, activation='relu', input_shape=(240, 6)))
#         model4.add(Conv1D(NConv2, 5, activation='relu'))
#         model4.add(GlobalMaxPooling1D())
#         model4.add(Dense(NDense1, activation='relu'))
#         model4.add(Dense(NDense2, activation='relu'))
#         model4.add(Dense(20, activation='softmax'))
#         model4.compile(loss='categorical_crossentropy', optimizer=nadam, metrics=['accuracy'])
#         model4.fit(xTrain,
#                    yTrain,
#                    epochs=epochs,
#                    validation_data=(xVal, yVal),
#                    callbacks=[early_stopping],
#                    verbose=1,
#                    batch_size=batch_size)

#         predictedTest = np.argmax(model4.predict(xTest), axis=1)
#         acc = accuracy_score(yTest, predictedTest)
#         precision, recall, f1, support = precision_recall_fscore_support(yTest, predictedTest, average='weighted')
#         acc_list.append(acc)
#         prec_list.append(precision)
#         recall_list.append(recall)
#         f1_list.append(f1)

#     acc_mean = np.mean(np.asarray(acc_list))
#     prec_mean = np.mean(np.asarray(prec_list))
#     recall_mean = np.mean(np.asarray(recall_list))
#     f1_mean = np.mean(np.asarray(f1_list))

#     acc_std = np.std(np.asarray(acc_list))
#     prec_std = np.std(np.asarray(prec_list))
#     recall_std = np.std(np.asarray(recall_list))
#     f1_std = np.std(np.asarray(f1_list))

#     my_file = file('stat_CONV_CONV_DENSE_DENSE.csv', 'a')
#     params = np.asarray([NConv1, NConv2, NDense1, NDense2, acc_mean, acc_std, prec_mean, prec_std, recall_mean, recall_std, f1_mean, f1_std])
#     np.savetxt(my_file, params.reshape((1, len(params))), fmt='%.2f')
#     my_file.close()


# # Model 5
# C1 = [76, 32, 78, 36, 35, 69, 70, 62, 72, 65]
# C2 = [53, 67, 47, 44, 68, 40, 41, 44, 78, 68]
# J1 = [69, 70, 59, 70, 36, 40, 54, 41, 78, 68]
# J2 = [36, 53, 54, 45, 60, 78, 46, 56, 71, 76]
# D1 = [53, 22, 50, 66, 52, 26, 18, 30, 61, 43]
# D2 = [31, 60, 45, 78, 49, 61, 30, 38, 58, 68]
# for NConv1, NConv2, NJanet1, NJanet2, NDense1, NDense2 in zip(C1, C2, J1, J2, D1, D2):
#     acc_list = []
#     prec_list = []
#     recall_list = []
#     f1_list = []
#     for i in range(repetition):
#         early_stopping = EarlyStopping(monitor='val_acc', patience=patience)
#         checkpoint = ModelCheckpoint('6dmg_try.h5', monitor='val_acc', verbose=0, save_weights_only=True)
#         nadam = Nadam(lr=1e-4, beta_1=0.9, beta_2=0.9)
#         model5 = Sequential()
#         model5.add(Conv1D(NConv1, 5, activation='relu', input_shape=(240, 6)))
#         model5.add(Dropout(0.2))
#         model5.add(Conv1D(NConv2, 5, activation='relu'))
#         model5.add(Dropout(0.2))
#         model5.add(JANET(NJanet1, activation='relu', return_sequences=True))
#         model5.add(Dropout(0.2))
#         model5.add(JANET(NJanet2, activation='relu'))
#         model5.add(Dropout(0.2))
#         model5.add(Dense(NDense1, activation='relu'))
#         model5.add(Dropout(0.2))
#         model5.add(Dense(NDense2, activation='relu'))
#         model5.add(Dropout(0.2))
#         model5.add(Dense(20, activation='softmax'))
#         model5.compile(loss='categorical_crossentropy', optimizer=nadam, metrics=['accuracy'])
#         model5.fit(xTrain,
#                    yTrain,
#                    epochs=epochs,
#                    validation_data=(xVal, yVal),
#                    callbacks=[early_stopping],
#                    verbose=1,
#                    batch_size=batch_size)

#         predictedTest = np.argmax(model5.predict(xTest), axis=1)
#         acc = accuracy_score(yTest, predictedTest)
#         precision, recall, f1, support = precision_recall_fscore_support(yTest, predictedTest, average='weighted')
#         acc_list.append(acc)
#         prec_list.append(precision)
#         recall_list.append(recall)
#         f1_list.append(f1)

#     acc_mean = np.mean(np.asarray(acc_list))
#     prec_mean = np.mean(np.asarray(prec_list))
#     recall_mean = np.mean(np.asarray(recall_list))
#     f1_mean = np.mean(np.asarray(f1_list))

#     acc_std = np.std(np.asarray(acc_list))
#     prec_std = np.std(np.asarray(prec_list))
#     recall_std = np.std(np.asarray(recall_list))
#     f1_std = np.std(np.asarray(f1_list))

#     my_file = file('stat_CONV_CONV_JANET_JANET_DENSE_DENSE_d2.csv', 'a')
#     params = np.asarray([NConv1, NConv2, NJanet1, NJanet2, NDense1, NDense2, acc_mean, acc_std, prec_mean, prec_std, recall_mean, recall_std, f1_mean, f1_std])
#     np.savetxt(my_file, params.reshape((1, len(params))), fmt='%.2f')
#     my_file.close()

# # Model 6
# C1 = [79, 76, 74, 63, 30, 31, 78, 63, 46, 71]
# C2 = [57, 78, 73, 47, 50, 78, 62, 69, 41, 33]
# J1 = [39, 28, 72, 48, 79, 72, 52, 78, 31, 28]
# J2 = [41, 63, 26, 59, 28, 76, 41, 50, 55, 42]
# D1 = [62, 72, 78, 30, 27, 64, 77, 15, 32, 66]
# D2 = [25, 50, 32, 46, 51, 41, 72, 76, 58, 31]
# for NConv1, NConv2, NJanet1, NJanet2, NDense1, NDense2 in zip(C1, C2, J1, J2, D1, D2):
#     acc_list = []
#     prec_list = []
#     recall_list = []
#     f1_list = []
#     for i in range(repetition):
#         early_stopping = EarlyStopping(monitor='val_acc', patience=patience)
#         checkpoint = ModelCheckpoint('6dmg_try.h5', monitor='val_acc', verbose=0, save_weights_only=True)
#         nadam = Nadam(lr=1e-4, beta_1=0.9, beta_2=0.9)
#         model6 = Sequential()
#         model6.add(Conv1D(NConv1, 5, activation='relu', input_shape=(240, 6)))
#         model6.add(Dropout(0.2))
#         model6.add(Conv1D(NConv2, 5, activation='relu'))
#         model6.add(Dropout(0.2))
#         model6.add(JANET(NJanet1, activation='relu', return_sequences=True))
#         model6.add(Dropout(0.2))
#         model6.add(JANET(NJanet2, activation='relu'))
#         model6.add(Dropout(0.2))
#         model6.add(Dense(NDense1, activation='relu'))
#         model6.add(Dropout(0.2))
#         model6.add(Dense(NDense2, activation='relu'))
#         model6.add(Dropout(0.2))
#         model6.add(Dense(20, activation='softmax'))
#         model6.compile(loss='categorical_crossentropy', optimizer=nadam, metrics=['accuracy'])
#         model6.fit(xTrain,
#                    yTrain,
#                    epochs=epochs,
#                    validation_data=(xVal, yVal),
#                    callbacks=[early_stopping],
#                    verbose=1,
#                    batch_size=batch_size)

#         predictedTest = np.argmax(model6.predict(xTest), axis=1)
#         acc = accuracy_score(yTest, predictedTest)
#         precision, recall, f1, support = precision_recall_fscore_support(yTest, predictedTest, average='weighted')
#         acc_list.append(acc)
#         prec_list.append(precision)
#         recall_list.append(recall)
#         f1_list.append(f1)

#     acc_mean = np.mean(np.asarray(acc_list))
#     prec_mean = np.mean(np.asarray(prec_list))
#     recall_mean = np.mean(np.asarray(recall_list))
#     f1_mean = np.mean(np.asarray(f1_list))

#     acc_std = np.std(np.asarray(acc_list))
#     prec_std = np.std(np.asarray(prec_list))
#     recall_std = np.std(np.asarray(recall_list))
#     f1_std = np.std(np.asarray(f1_list))

#     my_file = file('stat_CONV_CONV_JANET_JANET_DENSE_DENSE_d5.csv', 'a')
#     params = np.asarray([NConv1, NConv2, NJanet1, NJanet2, NDense1, NDense2, acc_mean, acc_std, prec_mean, prec_std, recall_mean, recall_std, f1_mean, f1_std])
#     np.savetxt(my_file, params.reshape((1, len(params))), fmt='%.2f')
#     my_file.close()


# # Model 7
# J1 = [52, 68, 56, 21, 24, 48, 41, 57, 76, 55]
# J2 = [39, 74, 74, 79, 73, 47, 61, 58, 51, 50]
# D1 = [61, 79, 61, 74, 34, 70, 43, 52, 55, 52]
# D2 = [48, 42, 41, 51, 58, 69, 52, 78, 66, 52]
# for NJanet1, NJanet2, NDense1, NDense2 in zip(J1, J2, D1, D2):
#     acc_list = []
#     prec_list = []
#     recall_list = []
#     f1_list = []
#     for i in range(repetition):
#         early_stopping = EarlyStopping(monitor='val_acc', patience=patience)
#         checkpoint = ModelCheckpoint('6dmg_try.h5', monitor='val_acc', verbose=0, save_weights_only=True)
#         nadam = Nadam(lr=1e-4, beta_1=0.9, beta_2=0.9)

#         model7 = Sequential()
#         model7.add(JANET(NJanet1, activation='relu', return_sequences=True, input_shape=(None, 6)))
#         model7.add(Dropout(0.2))
#         model7.add(JANET(NJanet2, activation='relu'))
#         model7.add(Dropout(0.2))
#         model7.add(Dense(NDense1, activation='relu'))
#         model7.add(Dropout(0.2))
#         model7.add(Dense(NDense2, activation='relu'))
#         model7.add(Dropout(0.2))
#         model7.add(Dense(20, activation='softmax'))
#         model7.compile(loss='categorical_crossentropy', optimizer=nadam, metrics=['accuracy'])
#         model7.fit(xTrain,
#                    yTrain,
#                    epochs=epochs,
#                    validation_data=(xVal, yVal),
#                    callbacks=[early_stopping],
#                    verbose=1,
#                    batch_size=batch_size)

#         predictedTest = np.argmax(model7.predict(xTest), axis=1)
#         acc = accuracy_score(yTest, predictedTest)
#         precision, recall, f1, support = precision_recall_fscore_support(yTest, predictedTest, average='weighted')
#         acc_list.append(acc)
#         prec_list.append(precision)
#         recall_list.append(recall)
#         f1_list.append(f1)

#     acc_mean = np.mean(np.asarray(acc_list))
#     prec_mean = np.mean(np.asarray(prec_list))
#     recall_mean = np.mean(np.asarray(recall_list))
#     f1_mean = np.mean(np.asarray(f1_list))

#     acc_std = np.std(np.asarray(acc_list))
#     prec_std = np.std(np.asarray(prec_list))
#     recall_std = np.std(np.asarray(recall_list))
#     f1_std = np.std(np.asarray(f1_list))

#     my_file = file('stat_JANET_JANET_DENSE_DENSE_d2.csv', 'a')
#     params = np.asarray([NJanet1, NJanet2, NDense1, NDense2, acc_mean, acc_std, prec_mean, prec_std, recall_mean, recall_std, f1_mean, f1_std])
#     np.savetxt(my_file, params.reshape((1, len(params))), fmt='%.2f')
#     my_file.close()
