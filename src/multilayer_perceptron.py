import numpy as np
from random import shuffle
from preprocessing import read_data, bag_of_words, sanitize
from sklearn.metrics import accuracy_score, classification_report


def tanh(x):
    return np.tanh(x)


def tanh_deriv(x):
    return 1 - np.tanh(x) ** 2


def mean_squared_error_deriv(y_true, y_pred):
    return y_pred - y_true


def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))


def sigmoid_deriv(x):
    return sigmoid(x)*(1-sigmoid(x))


def leaky_relu(x):
    return np.log(1 + np.exp(x))


def leaky_relu_deriv(x):
    return sigmoid(x)


def predict(instance, weights, biases, activation_fn=tanh):
    activation = instance.reshape((-1, 1))
    for layer_weights, layer_biases in zip(weights, biases):
        weighted_input = np.matmul(layer_weights, activation) + layer_biases
        activation = activation_fn(weighted_input)
    return activation


def predict_all(instances, weights, biases, activation_fn=tanh):
    return list(map(lambda x: predict(x, weights, biases, activation_fn), instances))


def backprop(instance, label, weights, biases,
             activation_fn=tanh, activation_fn_deriv=tanh_deriv,
             loss_deriv_fn=mean_squared_error_deriv):
    bias_grads = [np.zeros(b.shape) for b in biases]
    weight_grads = [np.zeros(w.shape) for w in weights]
    assert len(bias_grads) == len(weight_grads)
    num_layers = len(bias_grads) + 1

    instance = instance.reshape((-1, 1))
    label = label.reshape((-1, 1))

    activation = instance
    activations = [instance]
    weighted_inputs = []
    for layer_weights, layer_biases in zip(weights, biases):
        weighted_input = np.matmul(layer_weights, activation) + layer_biases
        weighted_inputs.append(weighted_input)
        activation = activation_fn(weighted_input)
        activations.append(activation)

    # this is where the math gets a bit hairy
    loss_deriv = loss_deriv_fn(label, activation)
    error = loss_deriv * activation_fn_deriv(weighted_input)

    weight_grads[-1] = np.matmul(error, activations[-2].transpose())
    bias_grads[-1] = error

    for i in range(2, num_layers):
        l = -i  # current layer
        weighted_input = weighted_inputs[l]
        activation_deriv = activation_fn_deriv(weighted_input)
        error = np.matmul(weights[l + 1].transpose(), error) * activation_deriv

        weight_grads[l] = np.matmul(error, activations[l - 1].transpose())
        bias_grads[l] = error

    return weight_grads, bias_grads


def weight_update(instances, labels, weights, biases, learning_rate,
                  activation_fn=tanh, activation_fn_deriv=tanh_deriv):
    weight_grads = [np.zeros(w.shape) for w in weights]
    bias_grads = [np.zeros(b.shape) for b in biases]
    assert len(bias_grads) == len(weight_grads)
    num_layers = len(bias_grads)

    assert len(instances) == len(labels)
    batch_len = len(instances)
    for instance, label in zip(instances, labels):
        weight_grads_delta, bias_grads_delta = \
            backprop(instance, label, weights, biases,
                     activation_fn=activation_fn,
                     activation_fn_deriv=activation_fn_deriv)
        for i in range(num_layers):
            weight_grads[i] += weight_grads_delta[i]
            bias_grads[i] += bias_grads_delta[i]

    for i in range(num_layers):
        weights[i] = weights[i] - (learning_rate / batch_len) * weight_grads[i]
        biases[i] = biases[i] - (learning_rate / batch_len) * bias_grads[i]


def stochastic_gradient_descent(instances, labels, weights, biases,
                                epochs=100, batch_size=16, learning_rate=0.01,
                                activation_fn=tanh,
                                activation_fn_deriv=tanh_deriv):
    assert len(instances) == len(labels)
    num_instances = len(instances)

    data = list(zip(instances, labels))
    for i in range(epochs):
        shuffle(data)
        batches = []
        for k in range(0, num_instances, batch_size):
            batches.append(data[k:k + batch_size])

        for batch in batches:
            batch_instances, batch_labels = zip(*batch)
            weight_update(
                batch_instances, batch_labels, weights, biases, learning_rate,
                activation_fn, activation_fn_deriv
            )

        print('Epoch ' + str(i) + ' complete.')
        predictions = np.argmax(predict_all(
            instances, weights, biases, activation_fn=activation_fn), axis=1)
        labels_test = np.argmax(labels, axis=1)
        print('Accuracy:', accuracy_score(labels_test, predictions))
        print(classification_report(labels_test, predictions))


def bows_to_numpy(bows):
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


def run_test():
    # loading and shuffling data and splitting into train/test sets
    instances, labels = read_data('../data/Tweets.csv')

    paired = list(zip(instances, labels))
    shuffle(paired)
    instances, labels = zip(*paired)

    bows = list(map(bag_of_words, map(sanitize, instances)))

    bows, _, _ = bows_to_numpy(
        bows)
    labels, _, _ = labels_to_numpy(
        labels)

    train_size = 1000
    bows_tr, labels_tr = bows[:train_size], labels[:train_size]
    bows_test, labels_test = bows[train_size:], labels[train_size:]

    sizes = [len(bows[0]), 15, 3]
    biases = [np.random.randn(s, 1) for s in sizes[1:]]
    weights = [np.random.randn(s_out, s_in)
               for s_in, s_out in zip(sizes[:-1], sizes[1:])]

    # learning weights on train set
    stochastic_gradient_descent(
        bows_tr, labels_tr, weights, biases, epochs=50,
        activation_fn=sigmoid, activation_fn_deriv=sigmoid_deriv)

    # evaluating classification accuracy using learned weights on the test set
    predictions = np.argmax(predict_all(bows_test, weights, biases), axis=1)
    labels_test = np.argmax(labels_test, axis=1)
    print('Accuracy:', accuracy_score(labels_test, predictions))
    print(classification_report(labels_test, predictions))


if __name__ == "__main__":
    run_test()
