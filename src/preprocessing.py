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

