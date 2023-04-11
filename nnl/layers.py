import numpy as np

class dense:
    def __init__(self, n_inputs:int, n_neurons:int):
        """
        A class for making Dense layers
        
        Attributes:
        -----------
        n_inputs : int
            The number of inputs in each batch
        n_neruons : int
            The number of neurons in the Dense layer

        Methods:
        ----------
        forward(input = np.ndarray or a list)
            Accepts numpy array or a list as input to
            forward propagate and store the new value in
            self.output.
        """

        self.weights = 0.01 * np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
    
    def forward(self, inputs:list):
        """
        Forward propagates input and stores it in ```self.output```
        """
        inputs = np.array(input)
        self.output = np.dot(inputs, self.weights) + self.biases
    
    def forward(self, inputs:np.ndarray):
        """
        Forward propagates input and stores it in ```self.output```
        """
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        # Gradient on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient values
        self.dinputs = np.dot(dvalues, self.weights.T)
