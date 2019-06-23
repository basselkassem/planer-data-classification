import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

def draw_softmax_classification(X, Y, weights, bais):
    step = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    feature1, feature2  = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))

    X_in = np.c_[feature1.ravel(), feature2.ravel()]
    Y_props = np.dot(X_in, weights) + bais
    Y_predict = np.argmax(Y_props, axis = 1)

    Y_predict = Y_predict.reshape(feature1.shape)
    plt.contourf(feature1, feature2 , Y_predict, cmap = plt.cm.spring, alpha = 0.7)
    plt.scatter(X[:, 0], X[:, 1], c = Y, s = 40, cmap = plt.cm.spring)
    plt.xlim(feature1.min(), feature1.max())
    plt.ylim(feature2 .min(), feature2 .max())
    plt.show()

def draw_nn_classification(X, Y, w1, b1, w2, b2):
    step = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    feature1, feature2  = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))

    X_in = np.c_[feature1.ravel(), feature2.ravel()]
    activations = np.maximum(0, np.dot(X_in, w1) + b1)
    Y_props = np.dot(activations, w2) + b2
    Y_predict = np.argmax(Y_props, axis = 1)

    Y_predict = Y_predict.reshape(feature1.shape)
    plt.contourf(feature1, feature2 , Y_predict, cmap = plt.cm.spring, alpha = 0.7)
    plt.scatter(X[:, 0], X[:, 1], c = Y, s = 40, cmap = plt.cm.spring)
    plt.xlim(feature1.min(), feature1.max())
    plt.ylim(feature2 .min(), feature2 .max())
    plt.show()