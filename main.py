from nnl import bundle as nn
# import numpy as np
from nnfs.datasets import spiral_data

X, y = spiral_data(samples=100, classes=3)

layer1 = nn.dense(2,3)
activation1 = nn.relu()
layer2 = nn.dense(3,3)
loss_activation = nn.activation_softmax_loss_activation_category()

layer1.forward(X)
activation1.forward(layer1.output)
layer2.forward(activation1.output)
loss = loss_activation.forward(layer2.output,y)

loss_function = nn.categorical_cross_entropy()

print("Loss: ", loss)

loss_activation.backward(loss_activation.output, y)
layer2.backward(loss_activation.dinputs)
activation1.backward(layer2.dinputs)
layer1.backward(activation1.dinputs)

print(f"\nlayer 1:\n\ndweights = {layer1.dweights}\n\ndbiases = {layer1.dbiases}\n")
print(f"\nlayer 2:\n\ndweights = {layer2.dweights}\n\ndbiases = {layer2.dbiases}")