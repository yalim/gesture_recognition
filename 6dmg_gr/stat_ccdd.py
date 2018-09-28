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

npzfile = np.load('GestureDatasetPadded.npz')
xTrain = npzfile['xTrain']
yTrain = npzfile['yTrain']
xVal = npzfile['xVal']
yVal = npzfile['yVal']
xTest = npzfile['xTest']
yTest = npzfile['yTest']


repetition = 1
epochs = 2
patience = 3
batch_size = 100

# Model 4
C1 = [59, 71, 30, 43, 76, 38, 39, 53, 79, 69]
C2 = [79, 66, 58, 50, 71, 22, 29, 48, 43, 57]
D1 = [52, 18, 46, 37, 23, 48, 31, 77, 48, 73]
D2 = [60, 51, 65, 53, 19, 48, 35, 50, 40, 37]
for NConv1, NConv2, NDense1, NDense2 in zip(C1, C2, D1, D2):
    acc_list = []
    prec_list = []
    recall_list = []
    f1_list = []
    for i in range(repetition):
        early_stopping = EarlyStopping(monitor='val_acc', patience=patience)
        checkpoint = ModelCheckpoint('6dmg_try.h5', monitor='val_acc', verbose=0, save_weights_only=True)
        nadam = Nadam(lr=1e-4, beta_1=0.9, beta_2=0.9)

        model4 = Sequential()
        model4.add(Conv1D(NConv1, 5, activation='relu', input_shape=(240, 6)))
        model4.add(Conv1D(NConv2, 5, activation='relu'))
        model4.add(GlobalMaxPooling1D())
        model4.add(Dense(NDense1, activation='relu'))
        model4.add(Dense(NDense2, activation='relu'))
        model4.add(Dense(20, activation='softmax'))
        model4.compile(loss='categorical_crossentropy', optimizer=nadam, metrics=['accuracy'])
        model4.fit(xTrain,
                   yTrain,
                   epochs=epochs,
                   validation_data=(xVal, yVal),
                   callbacks=[early_stopping],
                   verbose=1,
                   batch_size=batch_size)

        predictedTest = np.argmax(model4.predict(xTest), axis=1)
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

    my_file = file('stat_CONV_CONV_DENSE_DENSE.csv', 'a')
    params = np.asarray([NConv1, NConv2, NDense1, NDense2, acc_mean, acc_std, prec_mean, prec_std, recall_mean, recall_std, f1_mean, f1_std])
    np.savetxt(my_file, params.reshape((1, len(params))), fmt='%.2f')
    my_file.close()

