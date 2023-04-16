import numpy as np

class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0
    
    def optimize(self, gradient):
        if self.m is None or self.m.shape != gradient.shape:
            self.m = np.zeros(gradient.shape)
            self.v = np.zeros(gradient.shape)
        
        #print("Gradient shape: ", gradient.shape)
        
        self.t += 1
        
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient ** 2)
        
        bias_corrected_m = self.m / (1 - self.beta1 ** self.t)
        bias_corrected_v = self.v / (1 - self.beta2 ** self.t)
        
        update = -self.learning_rate * bias_corrected_m / (np.sqrt(bias_corrected_v) + self.epsilon)
        
        #print("update shape: ", update.shape)
        
        return update

class SGD:
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate
    
    def update_params(self, layer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases
