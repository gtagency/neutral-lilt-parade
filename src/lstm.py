from keras.models import Sequential
from keras.layers import LSTM, Embedding, Dense
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from preprocessing import read_data, labels_to_numpy
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

if __name__ == "__main__":
    # Get the untransformed data
    X, y = read_data('../data/Tweets.csv')

    # Label each of the words in the data
    num_words = 8000
    t = Tokenizer(num_words=num_words)
    t.fit_on_texts(X)

    # Convert the data into labeled sequences of fixed length
    X = t.texts_to_sequences(X)
    X = pad_sequences(X)

    y, _, _ = labels_to_numpy(y)

    # Split into training and testing data
    train_percent = 0.5
    train_size = int(len(X) * train_percent)
    X_test = X[train_size:]
    y_test = y[train_size:]
    X = X[:train_size]
    y = y[:train_size]

    # Build the model
    model = Sequential() 
    model.add(Embedding(num_words or np.max(X) + 1, output_dim=256))
    model.add(LSTM(128))
    model.add(Dense(len(y[0]), activation='softmax'))
    model.compile(optimizer=Adam(), loss=categorical_crossentropy)

    # Evaluate the model
    y_test = np.argmax(y_test, axis=1)
    for i in range(10):
        model.fit(X, y, epochs=1, batch_size=2 ** i)

        pred = np.argmax(model.predict(X_test), axis=1)
        print(classification_report(y_true=y_test, y_pred=pred))
        print(accuracy_score(y_true=y_test, y_pred=pred))