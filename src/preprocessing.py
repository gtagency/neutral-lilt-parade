import pandas as pd 

def read_data(filename, instance='text', label='airline_sentiment'):
    df = pd.read_csv(filename)
    labels = df[label].values
    instances = df[instance].values
    return (instances, labels)

instances, labels = read_data('../data/Tweets.csv')
for i in range(10):
    print(instances[i], labels[i])