from __future__ import print_function

from preprocessing import read_data, bag_of_words, sanitize, bows_to_numpy, labels_to_numpy
from random import shuffle
from sklearn.metrics import accuracy_score, classification_report
import numpy as np


def euclidean(a, b):
    return np.sum((np.array(a) - np.array(b)) ** 2) ** 0.5


count = 0


def predict(instance, instances_train, labels_train, k=3, dist=euclidean):
    global count
    count += 1
    print(count)

    def key(x): return dist(x[0], instance)
    nearest = sorted(list(zip(instances_train, labels_train)), key=key)[:k]
    nearest_labels = list(zip(*nearest))[1]
    return np.argmax(np.sum(nearest_labels, axis=0))


def run_test():
    # loading and shuffling data and splitting into train/test sets
    instances, labels = read_data('../data/Tweets.csv')

    paired = list(zip(instances, labels))
    shuffle(paired)
    instances, labels = zip(*paired)

    bows = list(map(bag_of_words, map(sanitize, instances)))

    bows, _, _ = bows_to_numpy(bows)
    labels, _, _ = labels_to_numpy(labels)

    train_size = 10000
    test_size = 100
    bows_tr, labels_tr = bows[:train_size], labels[:train_size]
    bows_test, labels_test = bows[train_size:train_size + test_size],\
        labels[train_size:train_size + test_size]

    # learning weights on train set
    predictions = list(
        map(lambda x: predict(x, bows_tr, labels_tr), bows_test))

    # evaluating classification accuracy using learned weights on the test set
    labels_test = np.argmax(labels_test, axis=1)
    print('Accuracy:', accuracy_score(labels_test, predictions))
    print(classification_report(labels_test, predictions))


if __name__ == "__main__":
    run_test()
