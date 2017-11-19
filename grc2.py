import numpy as np
import matplotlib.pyplot as plt
import itertools


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    Print and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    acc = np.trace(cm) / np.sum(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
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
    plt.title('Confusion Matrix - Accuracy: ' + str(acc))


if __name__ == '__main__':
    a = np.asarray([[6., 0., 1., 0., 0., 0., 3., 0., 0.],
                    [1., 8., 0., 0., 0., 0., 0., 1., 0.],
                    [5., 0., 1., 0., 0., 0., 0., 4., 0.],
                    [0., 0., 5., 4., 0., 0., 0., 0., 1.],
                    [0., 0., 2., 1., 3., 0., 0., 4., 0.],
                    [0., 0., 0., 0., 0., 4., 2., 2., 2.],
                    [1., 0., 0., 0., 0., 0., 9., 0., 0.],
                    [0., 2., 0., 0., 0., 0., 0., 8., 0.],
                    [0., 1., 0., 0., 0., 0., 2., 4., 3.]])
    a_noise = np.asarray([[1630., 0.], [20., 1637.]])
    gest_list = ['CCW Circle', 'CW Circle', 'Move Up', 'Move Down', 'Move Right', 'Move Left', 'CCW Triangle', 'CW Triangle', 'Letter Z']
    g_vs_n = ['Noise', 'Gesture']
    plot_confusion_matrix(a_noise, g_vs_n)
    plt.show()

# [[ 6.  0.  1.  0.  0.  0.  3.  0.  0.],
#  [ 1.  8.  0.  0.  0.  0.  0.  1.  0.],
#  [ 5.  0.  1.  0.  0.  0.  0.  4.  0.],
#  [ 0.  0.  5.  4.  0.  0.  0.  0.  1.],
#  [ 0.  0.  2.  1.  3.  0.  0.  4.  0.],
#  [ 0.  0.  0.  0.  0.  4.  2.  2.  2.],
#  [ 1.  0.  0.  0.  0.  0.  9.  0.  0.],
#  [ 0.  2.  0.  0.  0.  0.  0.  8.  0.],
#  [ 0.  1.  0.  0.  0.  0.  2.  4.  3.]]

# [[1630    0]
#  [  20 1637]]

