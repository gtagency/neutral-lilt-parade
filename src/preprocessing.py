import pandas as pd 

def bag_of_words(text):
    return Counter(text.split())

def read_data(filename, instance='text', label='airline_sentiment'):
    df = pd.read_csv(filename)
    labels = df[label].values
    instances = df[instance].values
    return (instances, labels)

instances, labels = read_data('../data/Tweets.csv')
for i in range(10):
    counts = bag_of_words(instances[i])
    print(instances[i], counts, labels[i])
