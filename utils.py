import numpy as np
import matplotlib.pyplot as plt
import itertools


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


class fifo_array():
    """
        A numpy ndarray that has a fixed length and works as FIFO
    """
    def __init__(self, max_length):
        self.max_len = max_length
        self.arr = np.zeros((1, self.max_len))

    def add_element(self, element):
        """
            Adds one element to the end
        """
        self.arr2 = np.append(self.arr, element)
        self.arr2 = np.delete(self.arr2, 0, 0)
        self.arr = np.reshape(self.arr2, self.arr.shape)

    def get_value(self):
        return self.arr

    def change_length(self, new_length):
        """
            This function adds zero to the end when increasing the length and
        removes from beginning when decreasing length
        """
        if self.max_len <= new_length:
            self.arr3 = np.zeros((1, new_length))
            self.arr3[0, 0:self.max_len] = self.arr
            self.arr = self.arr3
            self.max_len = new_length
        else:
            x = [y for y in range(self.max_len - new_length)]
            print x
            self.arr4 = np.delete(self.arr, x, 1)
            self.arr = self.arr4
            self.max_len = new_length

if __name__ == '__main__':
    x = fifo_array(10)
    for a in range(20):
        x.add_element(a)
        print x.get_value()
