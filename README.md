# Neural-Networks-Library
A python library for making simple neural networks built with numpy

## Example of using NNL:
```python
from nnl import layers,activation
import numpy as np


inputs = np.array([[1.2,3.9],
                [4.3,6.1],
                [8.3,5.4]])

layer1 = layers.dense(2,10)
activation1 = activation.relu()
layer2 = layers.dense(10,3)
activation2 = activation.softmax()

layer1.forward(inputs)
activation1.forward(layer1.output)
layer2.forward(activation1.output)
activation2.forward(layer2.output)

print(layer1.output)
print('\nOutput after applying ReLu function:\n')
print(activation1.output)
print()
print(layer2.output)
print('\nOutput after applying Softmax function:\n')
print(activation2.output)
```
