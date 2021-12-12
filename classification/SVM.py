import numpy as np
import random


class SVMModel(object):
    """
    SVM model
    """

    def __init__(self, max_iter=10000, C=0.1, epsilon=0.00001):
        self.max_iter = max_iter
        self.C = C
        self.epsilon = epsilon
        self.alpha = None

    def fit(self, X, Y):
        """
        Training model
        :param X: shape = num_train, dim_feature
        :param Y: shape = num_train
        :return: loss_history
        """
        Y[Y == 0] = -1
        n, d = X.shape[0], X.shape[1]
        self.alpha = np.zeros(n)
        # Iteration
        for i in range(self.max_iter):
            diff = self._iteration(X, Y)
            if i % 1 == 0:
                print('Iter %r / %r, Diff %r' % (i, self.max_iter, diff))
            if diff < self.epsilon:
                break

    def predict(self, X):
        return np.sign(np.dot(self.w.T, X.T) + self.b).astype(int)

    def _iteration(self, X, Y):
        alpha = self.alpha
        alpha_prev = np.copy(alpha)
        n = alpha.shape[0]
        for j in range(n):
            # Find i not equal to j randomly
            i = j
            while i == j:
                i = random.randint(0, n - 1)
            x_i, x_j, y_i, y_j = X[i, :], X[j, :], Y[i], Y[j]
            # Define the similarity of instances. K11 + K22 - 2K12
            k_ij = np.dot(x_i, x_i.T) + np.dot(x_j, x_j.T) - 2 * np.dot(x_i, x_j.T)
            if k_ij == 0:
                continue
            a_i, a_j = alpha[i], alpha[j]
            # Calculate the boundary of alpha
            L, H = self._cal_L_H(self.C, a_j, a_i, y_j, y_i)
            # Calculate model parameters
            self.w = np.dot(X.T, np.multiply(alpha, Y))
            self.b = np.mean(Y - np.dot(self.w.T, X.T))
            # Iterate alpha_j and alpha_i according to 'Delta W(a_j)'
            E_i = self.predict(x_i) - y_i
            E_j = self.predict(x_j) - y_j
            alpha[j] = a_j + (y_j * (E_i - E_j) * 1.0) / k_ij
            alpha[j] = min(H, max(L, alpha[j]))
            alpha[i] = a_i + y_i * y_j * (a_j - alpha[j])
        diff = np.linalg.norm(alpha - alpha_prev)
        return diff

    def _cal_L_H(self, C, a_j, a_i, y_j, y_i):
        if y_i != y_j:
            L = max(0, a_j - a_i)
            H = min(C, C - a_i + a_j)
        else:
            L = max(0, a_i + a_j - C)
            H = min(C, a_i + a_j)
        return L, H

    def x2(self, x1):
        '''
        When dimension of features is 2, given x1 to calculate x2. (To plot in 2D)
        '''
        return (-self.b - x1 * self.w[0]) / self.w[1]
