"""
Activation functions
--------------------
List of available activation functions:
- Softmax
- ReLu
- ELU
- Sigmoid
"""

import numpy as np

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