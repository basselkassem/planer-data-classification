import numpy as np

class NeuralNetwork(object):
    def __init__(self, X, Y, classes_number, nerons_number = 100):
        self.X = X
        self.Y = Y
        self.samples_number = X.shape[0]
        self.features_number = X.shape[1]
        self.classes_number = classes_number
        
        self.w1 = np.random.randn(self.features_number, nerons_number)
        self.b1 = np.zeros((1, nerons_number))
        self.w2 = np.random.randn(nerons_number, classes_number)
        self.b2 = np.zeros((1, classes_number))

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
        self.hidden_layer_activation= np.maximum(0, np.dot(self.X, self.w1) + self.b1)
        scores = np.dot(self.hidden_layer_activation, self.w2) + self.b2
        scores_exp = np.exp(scores)
        self.probs = scores_exp / np.sum(scores_exp, axis = 1, keepdims = True)
        log_probs = -np.log(self.probs)
        log_probs = log_probs[range(self.samples_number), self.Y]
        regulariztion = self.reg_term / 2 * (np.sum(self.w1 ** 2) / np.sum(self.w2 ** 2))
        self.loss = np.sum(log_probs) / self.samples_number + regulariztion
 
    def compute_gradients(self):
        derv_scores = self.probs
        derv_scores[range(self.samples_number), self.Y] -= 1
        derv_scores /= self.samples_number

        self.dw2 = np.dot(self.hidden_layer_activation.T, derv_scores)
        self.db2 = np.sum(derv_scores, axis = 0, keepdims = True) 

        derv_hidden = np.dot(derv_scores, self.w2.T)
        derv_hidden[self.hidden_layer_activation <= 0] = 0
     
        self.dw1 = np.dot(self.X.T, derv_hidden)
        self.db1 = np.sum(derv_hidden, axis = 0, keepdims = True)
        self.dw2 += self.reg_term * self.w2
        self.dw1 += self.reg_term * self.w1
    
    def update_parameters(self):
        self.w1 += -self.learning_rate * self.dw1
        self.b1 += -self.learning_rate * self.db1
        self.w2 += -self.learning_rate * self.dw2
        self.b2 += -self.learning_rate * self.db2

    def evaluate(self):
        hidden_layer_activation = np.maximum(0, np.dot(self.X, self.w1) + self.b1)
        scores = np.dot(hidden_layer_activation, self.w2) + self.b2
        predicted_class = np.argmax(scores, axis = 1)
        print('training accuracy: {}'.format(np.mean(predicted_class == self.Y)))