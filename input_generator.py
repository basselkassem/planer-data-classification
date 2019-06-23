import numpy as np

def spiral_distribution(samples_num, features_num, classes_num):
    samples_num_per_class = int(samples_num / classes_num)
    X = np.zeros((samples_num, features_num))
    Y = np.zeros(samples_num, dtype = 'uint8')

    for j in range(classes_num):
        xi = range(samples_num_per_class * j, samples_num_per_class * (j + 1))
        raduis = np.linspace(0, 1, samples_num_per_class)
        theta = np.linspace(j * 4, (j + 1) * 4, samples_num_per_class) + \
            np.random.randn(samples_num_per_class) * 0.2
        X[xi] = np.c_[raduis * np.sin(theta), raduis * np.cos(theta)]
        Y[xi] = j
    return (X, Y)