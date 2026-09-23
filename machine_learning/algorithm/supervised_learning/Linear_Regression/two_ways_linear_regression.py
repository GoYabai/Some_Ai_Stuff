import numpy as np


def batch_gradient_descent_linear(X, y, lr=0.01, epochs=1000):
    m, n = X.shape
    theta = np.zeros(n)

    for epoch in range(epochs):
        y_hat = np.dot(X, theta)
        loss = (y_hat - y)**2
        print(loss)
        gradient = np.dot(X.T, (y_hat - y)) / m
        theta -= lr * gradient

    return theta

def stochastic_gradient_descent_linear(X, y, learning_rate=0.01, epochs=1000):
    m, n = X.shape
    theta = np.zeros(n)

    for epoch in range(epochs):
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(m):
            xi = X_shuffled[i]
            yi = y_shuffled[i]
            y_hat = np.dot(xi, theta)
            loss = (y_hat - yi)**2
            print(loss)

            gradient = (y_hat - yi) * xi
            theta -= learning_rate * gradient
    return theta

