from preprocessing import read_data
from sklearn.linear_model import SGDClassifier
from evaluation import evaluate_bow_classifier

if __name__ == "__main__":
    clf = SGDClassifier()
    instances, labels = read_data('../data/Tweets.csv')
    evaluate_bow_classifier(instances, labels, clf)
