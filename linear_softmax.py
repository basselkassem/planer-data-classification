import numpy as np

class LinearSoftMax(object):
    def __init__(self, X, Y, classes_number):
        self.X = X
        self.Y = Y
        self.samples_number = X.shape[0]
        self.features_number = X.shape[1]
        self.classes_number = classes_number
        self.weights = np.random.randn(self.features_number, classes_number)
        self.bias = np.zeros((1, classes_number))

    def train(self, learning_rate, reg_term, iteration_number):
        self.learning_rate = learning_rate
        self.reg_term = reg_term
        for i in range(iteration_number):
            self.compute_loss()
            self.compute_gradients()
            self.update_parameters()
            if i % 10 == 0:
                print("iteration {}: loss {}".format(i, self.loss))

    def compute_loss(self):
        scores = np.dot(self.X, self.weights) + self.bias
        scores_exp = np.exp(scores)
        self.probs = scores_exp / np.sum(scores_exp, axis = 1, keepdims = True)
        log_probs = -np.log(self.probs)
        log_probs = log_probs[range(self.samples_number), self.Y]
        regulariztion = self.reg_term * np.sum(self.weights ** 2) / 2
        self.loss = np.sum(log_probs) / self.samples_number + regulariztion
 
    def compute_gradients(self):
        derv_scores = self.probs
        derv_scores[range(self.samples_number), self.Y] -= 1
        derv_scores /= self.samples_number
        self.derv_weighs = np.dot(self.X.T, derv_scores)
        self.derv_weighs += self.weights * self.reg_term
        self.derv_bias = np.sum(derv_scores, axis = 0, keepdims = True)
    
    def update_parameters(self):
        self.weights += -self.learning_rate * self.derv_weighs
        self.bias += -self.learning_rate * self.derv_bias

    def evaluate(self):
        scores = np.dot(self.X, self.weights) + self.bias
        predicted_class = np.argmax(scores, axis = 1)
        print('training accuracy: {}'.format(np.mean(predicted_class == self.Y)))