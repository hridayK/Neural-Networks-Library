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
        self.inputs = np.array(input)
        self.output = np.dot(inputs, self.weights) + self.biases
    
    def forward(self, inputs:np.ndarray):
        """
        Forward propagates input and stores it in ```self.output```
        """
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        # Gradient on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient values
        self.dinputs = np.dot(dvalues, self.weights.T)

class softmax:
    """
    Softmax functions: converts a vector into probability distribution as output:\n
    ``self.output = e^zi / Σ (e^z)``\n
    where:
    zi = element of a batch z at index i
    """
    def forward(self, inputs:np.ndarray):
        """
        ```self.output``` gets values that are output of the Softmax function.
        """
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in \
            enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1,1)
            jacobian_matrix = np.diagflat(single_output) - \
            np.dot(single_output,single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


class relu:
    """
    ReLu: Rectied Linear Activation Unit\n
    Applies ReLu function giving the output as:
        ```max(0,inputs)```
    """
    def forward(self, inputs:np.ndarray):
        """
        ```self.output``` gets values that are output of the ReLu function.
        """
        self.inputs = inputs
        self.output = np.maximum(0,inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class elu:
    """
    ELU: Exponential Linear Unit
    It is a modified form of ReLu function giving output as:
    ```
    def elu(x):
        output = x if x > 0
        output = α * (e^x - 1) if x <= 0
    ```
    alpha(α) has default value as 1.0.
    """
    def forward(self, inputs:np.ndarray, alpha=1.0):
        """
        ```self.output``` gets values that are output of the ELU function.
        """
        self.output = np.where(input>0, inputs, alpha*(np.exp(inputs) - 1))

class sigmoid:
    """
    Sigmoid activation function gives outputs as:
    ```
        sigmoid(x) = 1 / (1 + exp(-x))
    ```
    """
    def forward(self, inputs):
        """
        ``self.output`` gets values that are output of the Sigmoid function.
        """
        self.output = 1 / (1 + np.exp(-1*inputs))

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
class categorical_cross_entropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape)==1:
            correct_confidences = y_pred_clipped[range(samples), y_true]

        elif len(y_true.shape)==2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])

        if(len(y_true.shape == 1)):
            y_true = np.eye(labels)[y_true]
        
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples

class activation_softmax_loss_activation_category():

    def __init__(self):
        self.activation =  softmax()
        self.loss = categorical_cross_entropy()
    
    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        if  len(y_true.shape)==2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] = -1
        self.dinputs = self.dinputs / samples