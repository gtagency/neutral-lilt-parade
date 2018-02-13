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