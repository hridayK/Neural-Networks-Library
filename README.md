# Neural-Networks-Library
A python library for making simple neural networks built with numpy

## Example of forward propogation:
```python
from nnl import layers
import numpy as np


inputs = np.array([[1.2,3.9],
                [4.3,6.1],
                [8.3,5.4]])

layer1 = layers.dense(2,10)
layer2 = layers.dense(10,3)

layer1.forward(inputs=inputs)
layer2.forward(inputs=layer1.output)

print(layer2.output)
```
