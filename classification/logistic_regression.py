import numpy as np


class RegressionModel(object):
    """
    Logistic Regression
    """

    def __init__(self, learning_rate=0.1, n_iter=10000):
        self.W = None
        self.learning_rate = learning_rate
        self.n_iter = n_iter

    def train(self, X, Y):
        """
        Model training
        :param X: shape = num_train, dim_feature
        :param Y: shape = num_train
        :return: loss_history
        """
        Y[Y == -1] = 0
        Y = Y[:, np.newaxis]
        num_train, dim_feature = X.shape
        # w * x + b
        x_train_ = np.hstack((X, np.ones((num_train, 1))))
        self.W = 0.001 * np.random.randn(dim_feature + 1, 1)
        loss_history = []
        for i in range(self.n_iter + 1):
            # linear transformation: w * x + b
            g = np.dot(x_train_, self.W)
            # sigmoid: 1 / (1 + e**-x)
            h = 1 / (1 + np.exp(-g))
            # cross entropy: 1/m * sum((y*np.log(h) + (1-y)*np.log((1-h))))
            loss = -np.sum(Y * np.log(h) + (1 - Y) * np.log(1 - h)) / num_train
            loss_history.append(loss)
            # dW = cross entropy' = 1/m * sum(h-y) * x
            dW = x_train_.T.dot(h - Y) / num_train
            # W = W - dW
            self.W -= self.learning_rate * dW
            # debug
            if i % 1000 == 0:
                print('Iters: %r/%r Loss: %r' % (i, self.n_iter, loss))
        return loss_history

    def x2(self, x1):
        '''
        When dimension of features is 2, given x1 to calculate x2. (To plot in 2D)
        '''
        return (-self.W[2, 0] - x1 * self.W[0, 0]) / self.W[1, 0]
