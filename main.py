import input_generator
import visualizer
import linear_softmax
import neural_network
import numpy as np

samples_number = 1000
features_number = 2
classes_number = 3
X, Y = input_generator.spiral_distribution(samples_number, features_number, classes_number)

softmax_classifier = linear_softmax.LinearSoftMax(X, Y, classes_number)
softmax_classifier.train(learning_rate = 1e-0, reg_term = 1e-3, iteration_number = 1000)
softmax_classifier.evaluate()
visualizer.draw_softmax_classification(X, Y, softmax_classifier.weights , softmax_classifier.bias)

nn_classifier = neural_network.NeuralNetwork(X, Y, classes_number)
nn_classifier.train(learning_rate = 1e-0, reg_term = 1e-3, iteration_number = 10000)
nn_classifier.evaluate()
visualizer.draw_nn_classification(X, Y, nn_classifier.w1, nn_classifier.b1,
                                    nn_classifier.w2 , nn_classifier.b2)