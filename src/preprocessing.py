import pandas as pd 
from collections import Counter

def bag_of_words(text):
    return Counter(text.split())

def sanitize(text):
    text = text.replace('.', ' ')
    text = text.replace('"', ' ')
    text= text.lower()
    return text

def predict(bow, weights, labels):
    scores = {}
    # Get the score for each label by computing the dot product of its 
    # corresponding weights and the bag of words
    for label in labels:
        score = 0
        for word in bow:
            score += bow[word] * weights[label][word]
        scores[label] = score
    
    best_label = labels[0]
    best_score = scores[best_label]
    # Get the best label and score
    for label in scores:
        if scores[label] > best_score:
            best_label = label
            best_score = scores[label]
    # Equivalent to: 
    #     best_label = max(scores, key=lambda label: scores[label])
    #     best_score = scores[label]
    return best_label, best_score

def read_data(filename, instance='text', label='airline_sentiment'):
    df = pd.read_csv(filename)
    labels = df[label].values
    instances = df[instance].values
    return (instances, labels)

# Example prediction: 
#     Service was bad. (negative)

# First, we clean up the sentence and get a bag of words representation.
instance = bag_of_words(sanitize('Service was bad.'))

# Then, we set up our prediction weights.
# The weights below assume that 'service' and 'was' are not very relevant
# to a positive classification, but the word 'bad' is very non-positive
positive_weights = {
    'service': 0.2,
    'was': 0.0,
    'bad': -1
}
# The weights below assume that 'service' and 'was' are pretty neutral
neutral_weights = {
    'service': 0.5,
    'was': 0.5,
    'bad': 0.1
}
# The weights below assume that 'bad' is very negative.
negative_weights = {
    'service': 0.1,
    'was': 0.3,
    'bad': 0.9
}

# Put all the weights together so we can index into them by their labels
all_weights = {
    'positive': positive_weights,
    'neutral': neutral_weights,
    'negative': negative_weights
}

# Get a list of labels from all of the unique keys (indexes) in the weights
labels = list(set(all_weights.keys()))

# Predict the label of the instance using the weights above and the list of
# possible labels
prediction = predict(instance, all_weights, labels)
print('Predicted label:', prediction[0])
print('Score:', prediction[1])

