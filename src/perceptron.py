from __future__ import print_function

from collections import defaultdict
from linear import predict, predict_all
from preprocessing import read_data, bag_of_words, sanitize
from random import shuffle
from sklearn.metrics import classification_report, accuracy_score


def perceptron_update(bow, label, weights, labels):
    update = defaultdict(lambda: defaultdict(float))
    prediction, _ = predict(bow, weights, labels)
    if prediction == label:  # no changes need to be made to the weights
        return update

    # for mean squared error, `w(t+1) = w(t) + learn_rate * (y - y_pred) * x`
    # Here, we are computing `(y - y_pred) * x` as `y * x - y_pred * x`
    for word in bow:
        update[label][word] += bow[word]
        update[prediction][word] -= bow[word]
    return update


def estimate_weights(bows, labels, n_iterations, learning_rate=0.01):
    label_set = list(set(labels))
    weights = defaultdict(lambda: defaultdict(float))
    for _ in range(n_iterations):
        # run the perceptron update for each instance and update the 
        # weights accordingly
        for bow, label in zip(bows, labels):
            update = perceptron_update(bow, label, weights, label_set)
            for l in update:
                for u in update[l]:
                    weights[l][u] += update[l][u] * learning_rate
    return weights

# Loads, trains on, predicts, and scores on the training data.
# TODO add more measures of classifier quality, add validation split
def run_test():
    # loading and shuffling data and splitting into train/test sets
    instances, labels = read_data('../data/Tweets.csv')

    paired = list(zip(instances, labels))
    shuffle(paired)
    instances, labels = zip(*paired)

    bows = list(map(bag_of_words, map(sanitize, instances)))
    bows_tr, labels_tr = bows[:10000], labels[:10000]
    bows_test, labels_test = bows[10000:], labels[10000:]

    # learning weights on train set
    weights = estimate_weights(bows_tr, labels_tr, 10)

    # evaluating classification accuracy using learned weights on the test set
    predictions = predict_all(bows_test, weights, list(set(labels)))
    labels_prediction = [p[0] for p in predictions]
    print('Accuracy:', accuracy_score(labels_test, labels_prediction))
    print(classification_report(labels_test, labels_prediction))


if __name__ == "__main__":
    run_test()
