import numpy as np
#import pandas as pd. Not used

#Activation function, the curve is not very effective.
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# output function
def softmax(logits):
    exponentials = np.exp(logits)
    return exponentials / np.sum(exponentials, axis=1).reshape(-1, 1)

# sigmoid,easy to derivation
def sigmoid_gradient(sigmoid):
    return np.multiply(sigmoid, (1 - sigmoid))

# Calculate the loss using softmax cross-entropy, and be sure to divide by the number of samples to get the average.
def loss(Y, y_hat):
    return -np.sum(Y * np.log(y_hat)) / Y.shape[0]

# Bias
def prepend_bias(X):
    return np.insert(X, 0, 1, axis=1)

# Prediction essentially involves passing through the network, specifically for training purposes.
def forward(X, w1, w2):
    h = sigmoid(np.matmul(prepend_bias(X), w1))
    y_hat = softmax(np.matmul(prepend_bias(h), w2))
    return (y_hat, h)

# Backpropagation
def back(X, Y, y_hat, w2, h):
    w2_gradient = np.matmul(prepend_bias(h).T, (y_hat - Y)) / X.shape[0]
    w1_gradient = np.matmul(prepend_bias(X).T, np.matmul(y_hat - Y, w2[1:].T)
                            * sigmoid_gradient(h)) / X.shape[0]
    return (w1_gradient, w2_gradient)

# direct prediction
def classify(X, w1, w2):
    y_hat, _ = forward(X, w1, w2)
    labels = np.argmax(y_hat, axis=1)
    return labels.reshape(-1, 1)

# Random initialization: Values cannot be the same, otherwise neurons will synchronize, and matrix operations won't be able to eliminate this symmetry.
def initialize_weights(n_input_variables, n_hidden_nodes, n_classes):
    w1_rows = n_input_variables + 1
    w1 = np.random.randn(w1_rows, n_hidden_nodes) * np.sqrt(1 / w1_rows)

    w2_rows = n_hidden_nodes + 1
    w2 = np.random.randn(w2_rows, n_classes) * np.sqrt(1 / w2_rows)

    return (w1, w2)

# Training with small batches is faster, but the performance ceiling isn't high. 
# In this case, a batch size of 600 is used, which is quite large, so the initial decrease in loss isn't very fast.
def prepare_batches(X_train, Y_train, batch_size):
    x_batches = []
    y_batches = []
    n_examples = X_train.shape[0]
    for batch in range(0, n_examples, batch_size):
        batch_end = batch + batch_size
        x_batches.append(X_train[batch:batch_end])
        y_batches.append(Y_train[batch:batch_end])
    return x_batches, y_batches

# report
def report(epoch, X_train, Y_train, X_test, Y_test, w1, w2):
    y_hat, _ = forward(X_train, w1, w2)
    training_loss = loss(Y_train, y_hat)
    classifications = classify(X_test, w1, w2)
    accuracy = np.average(classifications == Y_test) * 100.0
    print("%5d > Loss: %.8f, Accuracy: %.2f%%" %
          (epoch, training_loss, accuracy))

# train
def train(X_train, Y_train, X_test, Y_test, n_hidden_nodes,
          epochs, batch_size, lr, print_every=10):
    n_input_variables = X_train.shape[1]
    n_classes = Y_train.shape[1]

    w1, w2 = initialize_weights(n_input_variables, n_hidden_nodes, n_classes)#There are only two layers here because using more layers 
    #for MNIST can lead to overfitting. Essentially, the features are too few, so adding some dropout might be necessary.
    #If you want to train on a dataset like CIFAR-10, you can simply add more layers. Since the activation function used is sigmoid, 
    # it's best to add batch normalization between each layer. You can just use Keras with a sequence like dense, normalization, dense, and so on.
    x_batches, y_batches = prepare_batches(X_train, Y_train, batch_size)
    report(0, X_train, Y_train, X_test, Y_test, w1, w2)
    for epoch in range(epochs):
        for batch in range(len(x_batches)):
            y_hat, h = forward(x_batches[batch], w1, w2)
            w1_gradient, w2_gradient = back(x_batches[batch], y_batches[batch],
                                            y_hat, w2, h)
            w1 = w1 - (w1_gradient * lr)
            w2 = w2 - (w2_gradient * lr)
        if (epoch + 1) % print_every == 0:
            report(epoch + 1, X_train, Y_train, X_test, Y_test, w1, w2)
    return (w1, w2)
