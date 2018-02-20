import numpy as np
from random import shuffle
from preprocessing import read_data, bag_of_words, sanitize
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA

def bows_to_numpy(bows):
    vocab = set()
    for text in bows:
        vocab.update(text)

    forward_mapping = {}
    reverse_mapping = {}
    for i, word in enumerate(vocab):
        forward_mapping[word] = i
        reverse_mapping[i] = word

    instance_len = len(vocab)
    num_instances = len(bows)
    result = np.zeros((num_instances, instance_len))

    for i, bow in enumerate(bows):
        for word in bow:
            result[i][forward_mapping[word]] = bow[word]

    return result, forward_mapping, reverse_mapping


def labels_to_numpy(labels):
    label_set = set(labels)

    forward_mapping = {}
    reverse_mapping = {}
    for i, word in enumerate(label_set):
        forward_mapping[word] = i
        reverse_mapping[i] = word

    result = np.zeros((len(labels), len(label_set)))
    for i, label in enumerate(labels):
        result[i][forward_mapping[label]] = 1

    return result, forward_mapping, reverse_mapping


def run_test():
    # loading and shuffling data and splitting into train/test sets
    instances, labels = read_data('../data/Tweets.csv')

    paired = list(zip(instances, labels))
    shuffle(paired)
    instances, labels = zip(*paired)

    bows = list(map(bag_of_words, map(sanitize, instances)))

    bows, _, _ = bows_to_numpy(bows)
    labels, _, _ = labels_to_numpy(labels)

    # pca = PCA(n_components=300)
    # bows = pca.fit_transform(bows)

    train_size = 10000
    bows_tr, labels_tr = bows[:train_size], labels[:train_size]
    bows_test, labels_test = bows[train_size:], labels[train_size:]

    # learning weights on train set
    clf = MLPClassifier(activation='tanh', verbose=2)
    clf.fit(bows_tr, labels_tr)

    # evaluating classification accuracy using learned weights on the test set
    predictions = np.argmax(clf.predict(bows_test), axis=1)
    labels_test = np.argmax(labels_test, axis=1)
    print('Accuracy:', accuracy_score(labels_test, predictions))
    print(classification_report(labels_test, predictions))


if __name__ == "__main__":
    run_test()
