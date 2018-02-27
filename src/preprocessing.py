import pandas as pd 
from collections import Counter

def bag_of_words(text):
    return Counter(text.split())

def sanitize(text):
    text = text.replace('.', ' ')
    text = text.replace('"', ' ')
    text= text.lower()
    return text

def read_data(filename, instance='text', label='airline_sentiment'):
    df = pd.read_csv(filename)
    labels = df[label].values
    instances = df[instance].values
    return (instances, labels)

def sanitize_all(texts):
    return [sanitize(text) for text in texts]

def bag_of_words_all(texts):
    return [bag_of_words(text) for text in texts]

def bows_to_numpy(bows):
    import numpy as np
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
    import numpy as np
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
