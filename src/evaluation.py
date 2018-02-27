from __future__ import print_function

from random import shuffle
from preprocessing import bag_of_words, sanitize, bows_to_numpy, labels_to_numpy
import numpy as np
from sklearn.metrics import accuracy_score, classification_report


def evaluate_classifier(instances, labels, clf, train_percent=0.5,
                        use_argmax_labels=True,
                        preprocess_instances=lambda X: (X, _, _),
                        preprocess_labels=lambda X: (X, _, _)):
    # shuffle the data
    paired = list(zip(instances, labels))
    shuffle(paired)
    instances, labels = zip(*paired)

    # convert to bows
    instances, _, _ = preprocess_instances(instances)
    labels, labels_forward, _ = preprocess_labels(labels)
    label_names, label_indices = zip(*(
        (k, labels_forward[k]) for k in labels_forward
    ))

    if use_argmax_labels:
        labels = np.argmax(labels, axis=1)

    train_size = int(len(labels) * train_percent)
    instances_tr, labels_tr = instances[:train_size], labels[:train_size]
    instances_test, labels_test = instances[train_size:], labels[train_size:]

    # learning weights on train set
    clf.fit(instances_tr, labels_tr)

    # evaluating classification accuracy using learned weights on the test set
    predictions = clf.predict(instances_test)

    if not use_argmax_labels:
        predictions = np.argmax(predictions, axis=1)
        labels_test = np.argmax(labels_test, axis=1)

    print('Accuracy:', accuracy_score(labels_test, predictions))
    print(classification_report(labels_test, predictions,
                                labels=label_indices, target_names=label_names))


def evaluate_bow_classifier(instances, labels, clf, train_percent=0.5,
                            use_argmax_labels=True, use_numpy_bows=True):
    def instances_to_bows(X):
        return list(map(bag_of_words, map(sanitize, X))), None, None

    def instances_to_numpy_bows(_instances):
        return bows_to_numpy(instances_to_bows(_instances)[0])[0], None, None

    preprocess_instances = instances_to_numpy_bows if use_numpy_bows \
        else instances_to_bows

    evaluate_classifier(
        instances, labels, clf, train_percent=train_percent,
        preprocess_instances=preprocess_instances,
        preprocess_labels=labels_to_numpy,
        use_argmax_labels=use_argmax_labels
    )
