from collections import Counter, defaultdict

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
