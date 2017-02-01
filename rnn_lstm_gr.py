import theano
import theano.tensor as T
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score
import time
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from six.moves import cPickle

# TODO: save and load weights


def init_weight(Mi, Mo):
    return np.random.randn(Mi, Mo) / np.sqrt(Mi + Mo)


class LSTM:
    def __init__(self, Mi, Mo, f):
        self.Mi = Mi
        self.Mo = Mo
        self.f  = f

        # numpy init
        Wxi = init_weight(Mi, Mo)
        Whi = init_weight(Mo, Mo)
        Wci = init_weight(Mo, Mo)
        bi  = np.zeros(Mo)
        Wxf = init_weight(Mi, Mo)
        Whf = init_weight(Mo, Mo)
        Wcf = init_weight(Mo, Mo)
        bf  = np.zeros(Mo)
        Wxc = init_weight(Mi, Mo)
        Whc = init_weight(Mo, Mo)
        bc  = np.zeros(Mo)
        Wxo = init_weight(Mi, Mo)
        Who = init_weight(Mo, Mo)
        Wco = init_weight(Mo, Mo)
        bo  = np.zeros(Mo)
        c0  = np.zeros(Mo)
        h0  = np.zeros(Mo)

        # theano vars
        self.Wxi = theano.shared(Wxi, 'Wxi')
        self.Whi = theano.shared(Whi, 'Whi')
        self.Wci = theano.shared(Wci, 'Wci')
        self.bi  = theano.shared(bi, 'bi')
        self.Wxf = theano.shared(Wxf, 'Wxf')
        self.Whf = theano.shared(Whf, 'Whf')
        self.Wcf = theano.shared(Wcf, 'Wcf')
        self.bf  = theano.shared(bf, 'bf')
        self.Wxc = theano.shared(Wxc, 'Wxc')
        self.Whc = theano.shared(Whc, 'Whc')
        self.bc  = theano.shared(bc, 'bc')
        self.Wxo = theano.shared(Wxo, 'Wxo')
        self.Who = theano.shared(Who, 'Who')
        self.Wco = theano.shared(Wco, 'Wco')
        self.bo  = theano.shared(bo, 'bo')
        self.c0  = theano.shared(c0, 'c0')
        self.h0  = theano.shared(h0, 'h0')
        self.params = [
            self.Wxi,
            self.Whi,
            self.Wci,
            self.bi,
            self.Wxf,
            self.Whf,
            self.Wcf,
            self.bf,
            self.Wxc,
            self.Whc,
            self.bc,
            self.Wxo,
            self.Who,
            self.Wco,
            self.bo,
            self.c0,
            self.h0,
        ]

    def recurrence(self, x_t, h_t1, c_t1):
        i_t = T.nnet.sigmoid(x_t.dot(self.Wxi) + h_t1.dot(self.Whi) + c_t1.dot(self.Wci) + self.bi)
        f_t = T.nnet.sigmoid(x_t.dot(self.Wxf) + h_t1.dot(self.Whf) + c_t1.dot(self.Wcf) + self.bf)
        c_t = f_t * c_t1 + i_t * T.tanh(x_t.dot(self.Wxc) + h_t1.dot(self.Whc) + self.bc)
        o_t = T.nnet.sigmoid(x_t.dot(self.Wxo) + h_t1.dot(self.Who) + c_t.dot(self.Wco) + self.bo)
        h_t = o_t * T.tanh(c_t)
        return h_t, c_t

    def output(self, x):
        # input X should be a matrix (2-D)
        # rows index time
        [h, c], _ = theano.scan(
            fn=self.recurrence,
            sequences=x,
            outputs_info=[self.h0, self.c0],
            n_steps=x.shape[0],
        )
        return h


class RNN:
    def __init__(self, hidden_layer_sizes):
        self.hidden_layer_sizes = hidden_layer_sizes

    def fit(self, X, Y, learning_rate=1e-6, epochs=1000,
            batch_size=10,
            activation=T.nnet.relu,
            momentum=0.99,
            temperature=None):
        N, D = X.shape
        print 'ND', N, D
        V = len(set(Y))

        self.hidden_layers = []
        Mi = D
        for Mo in self.hidden_layer_sizes:
            lstm = LSTM(Mi, Mo, activation)
            self.hidden_layers.append(lstm)
            Mi = Mo

        Wo = init_weight(Mi, V)
        bo = np.zeros(V)

        self.Wo = theano.shared(Wo)
        self.bo = theano.shared(bo)

        self.params = [self.Wo, self.bo]
        for ru in self.hidden_layers:
            self.params += ru.params

        thX = T.matrix('X')
        thY = T.ivector('Y')
        pY = self.forward(thX)

        cost = -T.mean(T.log(pY[T.arange(thY.shape[0]), thY]))
        prediction = self.predict(thX)
        grads = T.grad(cost, self.params)

        updates = [(p, p - learning_rate*g) for p, g in zip(self.params, grads)]

        train_op = theano.function(inputs=[thX, thY],
                                   outputs=[cost, prediction],
                                   updates=updates,
                                   allow_input_downcast=True)

        n_batches = N / batch_size

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

    def forward(self, X):
        for h in self.hidden_layers:
            X = h.output(X)
        return T.nnet.softmax(X.dot(self.Wo) + self.bo)

    def predict(self, X):
        y_hat = self.forward(X)
        return T.argmax(y_hat, axis=1)

    # def save(self, filename):
    #     np.savez(filename, *p.get_value() for p in self.params)

    # def set(self):
    #     pass

    # def load(self):
    #     pass

if __name__ == '__main__':
    x = np.genfromtxt('./GestureDataset_Padded.csv', delimiter=',')

    yis = x[:, 0] - 1
    print len(set(yis))
    print len(yis)
    xis = x[:, 2:]

    xis, yis = shuffle(xis, yis)

    x_train = xis[:800]
    y_train = yis[:800]
    x_test = xis[801:]
    y_test = yis[801:]

    start_time = time.time()

    rnn = RNN([200])
    rnn.fit(x_train, y_train, epochs=10)
    print("--- %s seconds ---" % (time.time() - start_time))

    print 'Training Completed.'

    f = open('lstm_rnn.save', 'wb')
    cPickle.dump(rnn, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
    print 'File saved!'

    f = open('lstm_rnn.save', 'rb')
    rnn_loaded = cPickle.load(f)
    f.close()
    print 'RNN loaded'

    x_test = theano.shared(x_test, 'x_test')
    predictions = rnn.predict(x_test)
    predictions_loaded = rnn_loaded.predict(x_test)
    print 'Predictions done!'
    x_train_test = theano.shared(x_train, 'x_train_test')
    predictions_train = rnn.predict(x_train_test)

    print '---TEST---'
    print (confusion_matrix(y_test, predictions.eval()))
    print '--'
    print confusion_matrix(y_test, predictions_loaded.eval())
    print 'Accuracy: ', accuracy_score(y_test, predictions.eval())
    print 'Recall: ', recall_score(y_test, predictions.eval())

    print '---TRAIN---'
    print (confusion_matrix(y_train, predictions_train.eval()))
    print 'Accuracy: ', accuracy_score(y_train, predictions_train.eval())
    print 'Recall: ', recall_score(y_train, predictions_train.eval())
    plt.show()
