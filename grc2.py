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


if __name__ == '__main__':
    a = np.asarray([[60, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 58, 0, 0, 0, 0, 2, 0, 0],
                [0, 0, 60, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 60, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 60, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 60, 0, 0, 0],
                [1, 0, 0, 0, 1, 0, 27, 0, 31],
                [0, 1, 0, 2, 0, 0, 1, 56, 0],
                [0, 0, 0, 0, 3, 0, 2, 0, 55]])
    gest_list = ['CCW Circle', 'CW Circle', 'Move Up', 'Move Down', 'Move Right', 'Move Left', 'CCW Triangle', 'CW Triangle', 'Letter Z']
    plot_confusion_matrix(a, gest_list)
    plt.show()
