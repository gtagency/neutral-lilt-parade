from preprocessing import bag_of_words, sanitize


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

def predict_all(bows, weights, labels):
    return [predict(bow, weights, labels) for bow in bows]

