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


# Model 7
J1 = [52, 68, 56, 21, 24, 48, 41, 57, 76, 55]
J2 = [39, 74, 74, 79, 73, 47, 61, 58, 51, 50]
D1 = [61, 79, 61, 74, 34, 70, 43, 52, 55, 52]
D2 = [48, 42, 41, 51, 58, 69, 52, 78, 66, 52]
for NJanet1, NJanet2, NDense1, NDense2 in zip(J1, J2, D1, D2):
    acc_list = []
    prec_list = []
    recall_list = []
    f1_list = []
    for i in range(repetition):
        early_stopping = EarlyStopping(monitor='val_acc', patience=patience)
        checkpoint = ModelCheckpoint('6dmg_try.h5', monitor='val_acc', verbose=0, save_weights_only=True)
        nadam = Nadam(lr=1e-4, beta_1=0.9, beta_2=0.9)

        model7 = Sequential()
        model7.add(JANET(NJanet1, activation='relu', return_sequences=True, input_shape=(None, 6)))
        model7.add(Dropout(0.2))
        model7.add(JANET(NJanet2, activation='relu'))
        model7.add(Dropout(0.2))
        model7.add(Dense(NDense1, activation='relu'))
        model7.add(Dropout(0.2))
        model7.add(Dense(NDense2, activation='relu'))
        model7.add(Dropout(0.2))
        model7.add(Dense(20, activation='softmax'))
        model7.compile(loss='categorical_crossentropy', optimizer=nadam, metrics=['accuracy'])
        model7.fit(xTrain,
                   yTrain,
                   epochs=epochs,
                   validation_data=(xVal, yVal),
                   callbacks=[early_stopping],
                   verbose=1,
                   batch_size=batch_size)

        predictedTest = np.argmax(model7.predict(xTest), axis=1)
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

    my_file = file('stat_JANET_JANET_DENSE_DENSE_d2.csv', 'a')
    params = np.asarray([NJanet1, NJanet2, NDense1, NDense2, acc_mean, acc_std, prec_mean, prec_std, recall_mean, recall_std, f1_mean, f1_std])
    np.savetxt(my_file, params.reshape((1, len(params))), fmt='%.2f')
    my_file.close()
