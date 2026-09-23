import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def batch_gradient_descent(X, y, learning_rate=0.01, epochs=1000):
    m, n = X.shape
    theta = np.zeros(n)

    for epoch in epochs:
        z = np.dot(X, theta)
        y_hat = sigmoid(z)

        loss = -y*np.log(y_hat) -(1-y)*np.log(1-y_hat)
        print(loss)

        gradient = np.dot(X.T, (y_hat - y)) / m
        theta -= learning_rate*gradient
    return theta

def stochastic_gradient_descent(X, y, learning_rate=0.01, epochs=1000):
    m, n = X.shape
    theta = np.zeros(n)
    
    for epoch in range(epochs):
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(m):
            xi = X_shuffled[i]
            yi = y_shuffled[i]

            z = np.dot(xi, theta)
            y_pred = sigmoid(z)

            loss = -yi*np.log(y_pred) -(1-yi)*np.log(1-y_pred)
            print(loss)

            gradient = np.dot(X.T, (y_pred - y))
            theta -= learning_rate * gradient
    return theta

