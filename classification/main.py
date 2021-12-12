import time

import matplotlib.pyplot as plt
from sklearn import datasets

from SVM import SVMModel
from logistic_regression import RegressionModel
from perceptron import Perceptron

# Prepare data
dataset = datasets.load_iris()
data = dataset.data[dataset.target <= 1][:, :2]
target = dataset.target[dataset.target <= 1]
label = dataset.target_names[:2]
# Plot data
plt.plot(data[:, 0][target == 0], data[:, 1][target == 0], 'oc', label=label[0])
plt.plot(data[:, 0][target == 1], data[:, 1][target == 1], 'o', c='gold', label=label[1])
times = []

# Perceptron
time_start = time.time()
perceptron = Perceptron(alpha=0.1, n_iter=10000)
perceptron.fit(data, target)
times.append(time.time() - time_start)
print("Perceptron", perceptron.W[0, 0], perceptron.W[1, 0], perceptron.intercept)
plt.plot([min(data[:, 0]), max(data[:, 0])], [perceptron.x2(min(data[:, 0])), perceptron.x2(max(data[:, 0]))],
         label='Perceptron')

# Logistic Regression
time_start = time.time()
regression = RegressionModel(0.1, 1000000)
regression.train(data, target)
times.append(time.time() - time_start)
print("Logistic Regression", regression.W[0, 0], regression.W[1, 0], regression.W[2, 0])
plt.plot([min(data[:, 0]), max(data[:, 0])], [regression.x2(min(data[:, 0])), regression.x2(max(data[:, 0]))],
         label='Regression')

# SVM
time_start = time.time()
SVM = SVMModel(10000)
SVM.fit(data, target)
times.append(time.time() - time_start)
print("SVM", SVM.w[0], SVM.w[1], SVM.b)
plt.plot([min(data[:, 0]), max(data[:, 0])], [SVM.x2(min(data[:, 0])), SVM.x2(max(data[:, 0]))],
         label='SVM')

plt.legend()
plt.savefig("classification_result.png")

print("time", times)
