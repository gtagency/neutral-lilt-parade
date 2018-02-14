from collections import Counter, defaultdict
from math import log
from linear import predict_all
from sklearn.metrics import accuracy_score
from preprocessing import read_data, bag_of_words, sanitize


def get_vocabulary(texts):
    vocab = set()
    for text in texts:
        vocab.update(text)
    return vocab


def get_label_word_counts(texts, labels, label):
    counter = Counter()
    for x, y in zip(texts, labels):
        if y == label:
            counter.update(x)
    return defaultdict(float, counter)


def estimate_prob_words_given_label(texts, labels, label, smoothing, vocab):
    # P(w | L) = P(w AND L) / P(L)
    conditional_probs = defaultdict(float)

    label_word_counts = get_label_word_counts(texts, labels, label)
    label_count = sum(label_word_counts.values()) + len(vocab) * smoothing
    for word in vocab:
        word_count = label_word_counts[word]
        conditional_probs[word] = log((word_count + smoothing) / label_count)
    return conditional_probs


def estimate_weights(texts, labels, smoothing):
    weights = defaultdict(lambda: defaultdict(float))

    label_set = set(labels)
    label_counts = Counter(labels)

    vocab = get_vocabulary(texts)

    for label in label_set:
        prob_words_given_label = estimate_prob_words_given_label(
            texts, labels, label, smoothing, vocab
        )
        for word in prob_words_given_label:
            weights[label][word] = prob_words_given_label[word]
        # TODO may add offset
        # weights[label]["**OFFSET**"] = label_counts[label] / len(labels)
    return weights


def find_best_smoother(texts_true, labels_true,
                       texts_val, labels_val,
                       smoothers):
    scores = {}
    label_set = set(labels_true).union(set(labels_val))
    for smoother in smoothers:
        weights = estimate_weights(texts_true, labels_true, smoother)
        predictions = predict_all(texts_val, weights, label_set)
        scores[smoother] = accuracy_score(labels_true, predictions)
    return max(smoothers, key=lambda s: scores[s])


def run_test():
    instances, labels = read_data('../data/Tweets.csv')
    weights = estimate_weights(instances, labels, 0.0001)
    predictions = predict_all(list(map(bag_of_words, instances)),
                              weights, list(set(labels)))
    prediction_labels = [p[0] for p in predictions]
    print(accuracy_score(labels, prediction_labels))


run_test()
