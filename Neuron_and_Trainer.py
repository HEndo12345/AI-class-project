import numpy as np
import random
import pickle as pkl
class Neuron:

    def __init__(self, dimension=1, weights=None, bias=None, activation=(lambda x: x)):

        self._dim = dimension
        self.w = weights or np.array([random.random()*(-1)**random.randint(0, 1) for _ in range(self._dim)])
        self.w = np.array([float(w) for w in self.w])
        self.b = bias if bias is not None else random.random()*(-1)**random.randint(0, 1)
        self.b = float(self.b)
        self._a = activation

    def __str__(self):

        return "Simple cell neuron\n\
        \tInput dimension: %d\n\
        \tBias: %f\n\
        \tWeights: %s\n\
        \tActivation: %s" % (self._dim, self.b, self.w, self._a.__name__)

    def __call__(self, x):

        yhat = self._a(np.dot(self.w, np.array(x)) + self.b)
        return yhat
    
    def predict(self, x):
        return self(x)

class Trainer:
    
    def __init__(self, dataset, model, loss):
        self.dataset = dataset
        self.model = model
        self.loss = loss
        
    def validate(self, data):
        
        results = [self.loss(self.model.predict(x), y) for x, y in data]
        return float(sum(result for result in results))/float(len(results))
    
    def accuracy(self, data):
        return 100*float(sum([1 for x, y in data if self.model.predict(x) == y]))/float(len(data))
    
    def train(self, lr, ne):
        print("training model on data...")
        average_loss = self.validate(self.dataset)
        accuracy = self.accuracy(self.dataset)
        print("initial average_loss: %.3f" % (average_loss))
        print("initial accuracy: %.3f" % (accuracy))
        for epoch in range(ne):
            for d in self.dataset:
                x, y = d
                x = np.array(x)
                yhat = self.model(x)
                error = y - yhat
                self.model.w += lr*(y-yhat)*x
                self.model.b += lr*(y-yhat)
            average_loss = self.validate(self.dataset)
            accuracy = self.accuracy(self.dataset)
            print('>epoch=%d, lrate=%.3f, average_loss=%.3f, accuracy=%.3f' % (epoch+1, lr, average_loss, accuracy))
        print("training complete")
        print("final average_loss: %.3f" % (average_loss))
        print("final accuracy: %.3f" % (self.accuracy(self.dataset)))

# activation functions

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))

def tanh(z):
    return np.tanh(z)

def relu(z):
    return np.maximum(z, 0)

def perceptron(z):
    return -1 if z<=0 else 1

# loss functions

def qloss(yhat, y):
    return (yhat-y)**2/2

def ploss(yhat, y):
    return max(0, -yhat*y)
