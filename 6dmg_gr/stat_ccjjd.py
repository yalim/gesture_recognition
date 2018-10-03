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

# Model 3
C1 = [71, 28, 78, 67, 60, 28, 11, 75, 29, 71]
C2 = [66, 68, 33, 59, 75, 58, 46, 51, 63, 78]
J1 = [43, 60, 77, 68, 42, 15, 39, 66, 43, 26]
J2 = [30, 27, 54, 41, 28, 25, 34, 42, 46, 50]
D = [44, 40, 45, 70, 20, 67, 36, 43, 65, 63]
for NConv1, NConv2, NJanet1, NJanet2, NDense in zip(C1, C2, J1, J2, D):
    acc_list = []
    prec_list = []
    recall_list = []
    f1_list = []
    for i in range(repetition):
        early_stopping = EarlyStopping(monitor='val_acc', patience=patience)
        checkpoint = ModelCheckpoint('6dmg_try.h5', monitor='val_acc', verbose=0, save_weights_only=True)
        nadam = Nadam(lr=1e-4, beta_1=0.9, beta_2=0.9)

        model3 = Sequential()
        model3.add(Conv1D(NConv1, 5, activation='relu', input_shape=(240, 6)))
        model3.add(Conv1D(NConv2, 5, activation='relu'))
        model3.add(JANET(NJanet1, activation='relu', return_sequences=True)) #, input_shape=(None, 6)))
        model3.add(JANET(NJanet2, activation='relu'))
        model3.add(Dense(NDense, activation='relu'))
        model3.add(Dense(20, activation='softmax'))
        model3.compile(loss='categorical_crossentropy', optimizer=nadam, metrics=['accuracy'])
        model3.fit(xTrain,
                   yTrain,
                   epochs=epochs,
                   validation_data=(xVal, yVal),
                   callbacks=[early_stopping],
                   verbose=1,
                   batch_size=batch_size)

        predictedTest = np.argmax(model3.predict(xTest), axis=1)
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

    my_file = file('stat_CONV_CONV_JANET_JANET_DENSE.csv', 'a')
    params = np.asarray([NConv1, NConv2, NJanet1, NJanet2, NDense, acc_mean, acc_std, prec_mean, prec_std, recall_mean, recall_std, f1_mean, f1_std])
    np.savetxt(my_file, params.reshape((1, len(params))), fmt='%.2f')
    my_file.close()