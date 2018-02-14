from collections import Counter, defaultdict
from math import log
from linear import predict_all, predict
from sklearn.metrics import accuracy_score, classification_report
from preprocessing import read_data, bag_of_words, sanitize, bag_of_words_all
from random import shuffle


# Gets the set of all words which appear in a collection of BOWs
def get_vocabulary(bows):
    vocab = set()
    for text in bows:
        vocab.update(text)
    return vocab


# Gets the amount of times each word shows up paired with a given label
def get_label_word_counts(bows, labels, label):
    counter = Counter()
    for x, y in zip(bows, labels):
        if y == label:
            counter.update(x)
    return defaultdict(float, counter)


# Gets the probabilities of each word given a particular label, taking into
# account smoothing
def estimate_prob_words_given_label(bows, labels, label, smoothing, vocab):
    # P(w | L) = P(w AND L) / P(L)
    conditional_probs = defaultdict(float)

    label_word_counts = get_label_word_counts(bows, labels, label)
    label_count = sum(label_word_counts.values()) + len(vocab) * smoothing
    for word in vocab:
        word_count = label_word_counts[word]
        conditional_probs[word] = log((word_count + smoothing) / label_count)
    return conditional_probs


# Estimates the weights as the probability of each word appearing given a label
# for all label pairs
def estimate_weights(bows, labels, smoothing):
    weights = defaultdict(lambda: defaultdict(float))

    label_set = set(labels)
    label_counts = Counter(labels)

    vocab = get_vocabulary(bows)

    for label in label_set:
        prob_words_given_label = estimate_prob_words_given_label(
            bows, labels, label, smoothing, vocab
        )
        for word in prob_words_given_label:
            weights[label][word] = prob_words_given_label[word]
    return weights


# Finds the best smoothing value given a list of candidates and a
# training/validation set
def find_best_smoother(texts_train, labels_train, texts_val, labels_val,
                       smoothers):
    scores = {}
    label_set = list(set(labels_train).union(set(labels_val)))
    for smoother in smoothers:
        weights = estimate_weights(texts_train, labels_train, smoother)
        predictions = [p[0] for p in predict_all(texts_val, weights, label_set)]
        scores[smoother] = accuracy_score(labels_val, predictions)
    return max(smoothers, key=lambda s: scores[s])


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
    weights = estimate_weights(bows_tr, labels_tr, 0.5)

    # evaluating classification accuracy using learned weights on the test set
    predictions = predict_all(bows_test, weights, list(set(labels)))
    prediction_labels = [p[0] for p in predictions]
    print('Accuracy:', accuracy_score(labels_test, prediction_labels))
    print(classification_report(labels_test, prediction_labels))
        


if __name__ == "__main__":
    run_test()
