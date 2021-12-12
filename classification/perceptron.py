from numpy import *


class Perceptron():
    """
    Perceptron model
    """

    def __init__(self, alpha=0.01, n_iter=20):
        self.alpha = alpha
        self.n_iter = n_iter

    def fit(self, X, Y):
        Y[Y == 0] = -1
        X = mat(X)
        m, n = shape(X)
        self.intercept = 0
        self.W = mat(ones((n, 1)))

        for i in range(self.n_iter):
            for x, y in zip(X, Y):
                y_ = float(x * self.W) + self.intercept
                if y * y_ < 0:
                    self.W += x.T * self.alpha * y
                    self.intercept += self.alpha * y
            if self.loss(X, Y) == 0:
                return i

    def loss(self, X, Y):
        sum_loss = 0.0
        for x, y in zip(X, Y):
            y_ = float(x * self.W) + self.intercept
            if y * y_ < 0:
                sum_loss += - y * y_
        return sum_loss

    def x2(self, x1):
        '''
        When dimension of features is 2, given x1 to calculate x2. (To plot in 2D)
        '''
        return (-self.intercept - x1 * self.W[0, 0]) / self.W[1, 0]
