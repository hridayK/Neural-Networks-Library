import numpy as np
class softmax:
    """
    Softmax functions: gives normalized values as output:\n
    ``self.output = e^zi / Σ (e^z)``\n
    where:
    zi = element of a batch z at index i
    """
    def forward(self, inputs:np.ndarray):
        """
        ```self.output``` get the values that are the output of the softmax function.
        """
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

class relu:
    """
    ReLu: Rectied Linear Activation Unit
        ```max(0,inputs)```
    """
    def forward(self, inputs:np.ndarray):
        """
        ```self.output``` get the values that are the output of the softmax function.
        """
        self.output = np.maximum(0,inputs)