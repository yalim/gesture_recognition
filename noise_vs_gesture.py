import theano
from theano import tensor as T
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score
import time
import csv


class HiddenLayer:
    def __init__(self, Mi, Mo, id_no):
        self.input = Mi
        self.output = Mo
        self.id = id_no
        self.W = theano.shared(np.random.randn(self.input, self.output), 'W_%s' % id_no)
        self.b = theano.shared(np.zeros(self.output), 'b_%s' % id_no)
        self.params = [self.W, self.b]

    def fwd(self, X):
        return T.nnet.relu(X.dot(self.W) + self.b)


class ANN:
    def __init__(self, hidden_layer_sizes):
        self.hidden_layers = hidden_layer_sizes
        self.hidden_layers_list = []

    def fit(self, X, Y, learning_rate=1e-3, batch_size=10, epochs=1000):
        Y = Y.astype(np.int32)
        X = X.astype(np.float32)
        n, f = X.shape
        d = len(set(Y))
        mi = f

        for ids, h in enumerate(self.hidden_layers):
            self.hidden_layers_list.append(HiddenLayer(mi, h, ids))
            mi = h

        self.W = theano.shared(np.random.randn(mi, d), 'W_op')
        self.b = theano.shared(np.zeros(d), 'b_op')

        self.params = [self.W, self.b]
        for h in self.hidden_layers_list:
            self.params += h.params

        thX = T.matrix('X')
        thY = T.ivector('Y')
        pY = self.forward(thX)

        cost = -T.mean(T.log(pY[T.arange(thY.shape[0]), thY]))
        prediction = self.predict(thX)
        grads = T.grad(cost, self.params)

        updates = [(p, p - learning_rate*g) for p, g in zip(self.params, grads)]

        train_op = theano.function(inputs=[thX, thY], outputs=[cost, prediction], updates=updates)

        n_batches = n / batch_size

        costs = []

        for i in xrange(epochs):
            print "Epoch: ", i
            X, Y = shuffle(X, Y)
            for j in xrange(n_batches):
                Xbatch = X[j * batch_size:(j * batch_size + batch_size)]
                Ybatch = Y[j * batch_size:(j * batch_size + batch_size)]

                c, p = train_op(Xbatch, Ybatch)
                costs.append(c)

        plt.plot(costs)
        plt.show()

    def forward(self, X):
        for h in self.hidden_layers_list:
            X = h.fwd(X)
        return T.nnet.softmax(X.dot(self.W) + self.b)

    def predict(self, X):
        y_hat = self.forward(X)
        return np.argmax(y_hat, axis=1)


if __name__ == '__main__':
    # Form the noise set
    x_acc_noise = []
    y_acc_noise = []
    z_acc_noise = []

    x_gyr_noise = []
    y_gyr_noise = []
    z_gyr_noise = []
    rr = 0
    with open('../python_denemeler/noisy_data.csv', 'r') as csvfile:
        sensors = csv.reader(csvfile, delimiter=',')
        for row in sensors:
            if rr > 0 and rr < 70001:
                sensors_data = map(float, row)
                x_acc_noise.append(sensors_data[1])
                y_acc_noise.append(sensors_data[2])
                z_acc_noise.append(sensors_data[3])

                x_gyr_noise.append(sensors_data[4])
                y_gyr_noise.append(sensors_data[5])
                z_gyr_noise.append(sensors_data[6])
            rr = rr + 1

    x_acc_noise = np.asarray(x_acc_noise)
    y_acc_noise = np.asarray(y_acc_noise)
    z_acc_noise = np.asarray(z_acc_noise)

    x_gyr_noise = np.asarray(x_gyr_noise)
    y_gyr_noise = np.asarray(y_gyr_noise)
    z_gyr_noise = np.asarray(z_gyr_noise)

    x_acc_noise = np.reshape(x_acc_noise, (700, 100))
    y_acc_noise = np.reshape(y_acc_noise, (700, 100))
    z_acc_noise = np.reshape(z_acc_noise, (700, 100))

    x_gyr_noise = np.reshape(x_gyr_noise, (700, 100))
    y_gyr_noise = np.reshape(y_gyr_noise, (700, 100))
    z_gyr_noise = np.reshape(z_gyr_noise, (700, 100))

    x_noise = np.hstack((x_acc_noise, y_acc_noise, z_acc_noise, x_gyr_noise, y_gyr_noise, z_gyr_noise))

    # with open('./GestureDataset_Padded.csv', 'r') as dataset:
    #     datas = csv.reader(dataset, delimiter=',')
    #     x = [map(float, xx) for xx in datas]

    # ##################################
    # x_np = np.asarray(x)
    # print x_np[0].shape
    # print x_np[0]

    x = np.genfromtxt('./GestureDataset_Padded.csv', delimiter=',')
    print x[:, 2:].shape
    print x_noise.shape

    yis = np.asarray(700*[0] + x.shape[0]*[1])
    xis = np.vstack((x_noise, x[:, 2:]))
    xis, yis = shuffle(xis, yis)
    x_train = xis[:600]
    y_train = yis[:600]
    x_test = xis[601:]
    y_test = yis[601:]

    start_time = time.time()

    ann2 = ANN([300])

    ann2.fit(x_train, y_train, epochs=150, learning_rate=1e-5)
    print 'Training Completed.'
    x_test = theano.shared(x_test, 'x_test')
    predictions = ann2.predict(x_test)
    print 'Predictions done!'

    print (confusion_matrix(y_test, predictions.eval()))
    plt.imshow(confusion_matrix(y_test, predictions.eval()))
    # plt.show()
    print 'Accuracy: ', accuracy_score(y_test, predictions.eval())
    print 'Recall: ', recall_score(y_test, predictions.eval())
    print("--- %s seconds ---" % (time.time() - start_time))
