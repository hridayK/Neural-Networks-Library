from nnl import bundle as nn
from nnl import optimizers
import numpy as np
from nnfs.datasets import spiral_data

# X, y = spiral_data(samples=100, classes=3)

# layer1 = nn.dense(2,3)
# activation1 = nn.relu()
# layer2 = nn.dense(3,3)
# loss_activation = nn.activation_softmax_loss_activation_category()

# layer1.forward(X)
# activation1.forward(layer1.output)
# layer2.forward(activation1.output)
# loss = loss_activation.forward(layer2.output,y)

# print("weights shape after forward: ", layer1.weights.shape)
# print("bias shape after forward: ", layer1.biases.shape)

# loss_function = nn.categorical_cross_entropy()

# print("Loss: ", loss)

# loss_activation.backward(loss_activation.output, y)
# layer2.backward(loss_activation.dinputs)
# activation1.backward(layer2.dinputs)
# layer1.backward(activation1.dinputs)

# print("weights shape after backward: ", layer1.dweights.shape)
# print("bias shape after backward: ", layer1.dbiases.shape)

# layer1.dweights = layer1.dweights.T
# layer1.dbiases = layer1.dbiases.T
# layer2.dweights = layer2.dweights.T
# layer2.dbiases = layer2.dbiases.T

# print("weights shape after transpose: ", layer1.dweights.shape)
# print("bias shape after transpose: ", layer1.dbiases.shape)
# print("weights shape after transpose: ", layer2.dweights.shape)
# print("bias shape after transpose: ", layer2.dbiases.shape)



# print(f"\nlayer 1:\n\ndweights = {layer1.dweights}\n\ndbiases = {layer1.dbiases}\n")
# print(f"\nlayer 2:\n\ndweights = {layer2.dweights}\n\ndbiases = {layer2.dbiases}")

# optimizer = optimizers.AdamOptimizer(learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8)

# print("weights shape: ", layer2.dweights.shape)

# layer1.dweights += optimizer.optimize(layer1.dweights)
# layer1.dbiases += optimizer.optimize(layer1.dbiases)
# layer2.dweights += optimizer.optimize(layer2.dweights)
# layer2.dbiases += optimizer.optimize(layer2.dbiases)

# print(f"\nlayer 1:\n\ndweights = {layer1.dweights}\n\ndbiases = {layer1.dbiases}\n")
# print(f"\nlayer 2:\n\ndweights = {layer2.dweights}\n\ndbiases = {layer2.dbiases}")

X, y = spiral_data(samples=100, classes=3)

layer1 = nn.dense(2, 64)
activation1 = nn.relu()
layer2 = nn.dense(64, 64)
activation2 = nn.relu()
layer3 = nn.dense(64, 64)
activation3 = nn.relu()
layer4 = nn.dense(64, 3)
loss_activation = nn.activation_softmax_loss_activation_category()

optimizer = optimizers.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)

# Train model
for epoch in range(1000):
    # Forward pass
    layer1.forward(X)
    activation1.forward(layer1.output)
    layer2.forward(activation1.output)
    activation2.forward(layer2.output)
    layer3.forward(activation2.output)
    activation3.forward(layer3.output)
    layer4.forward(activation3.output)
    loss = loss_activation.forward(layer4.output, y)
    
    # Print loss every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}")
    
    # Backward pass
    loss_activation.backward(loss_activation.output, y)
    layer4.backward(loss_activation.dinputs)
    activation3.backward(layer4.dinputs)
    layer3.backward(activation3.dinputs)
    activation2.backward(layer3.dinputs)
    layer2.backward(activation2.dinputs)
    activation1.backward(layer2.dinputs)
    layer1.backward(activation1.dinputs)
    
    # Updating model parameters
    layer1.weights += optimizer.optimize(layer1.dweights)
    layer1.biases += optimizer.optimize(layer1.dbiases)
    layer2.weights += optimizer.optimize(layer2.dweights)
    layer2.biases += optimizer.optimize(layer2.dbiases)
    layer3.weights += optimizer.optimize(layer3.dweights)
    layer3.biases += optimizer.optimize(layer3.dbiases)
    layer4.weights += optimizer.optimize(layer4.dweights)
    layer4.biases += optimizer.optimize(layer4.dbiases)

# Evaluate model
layer1.forward(X)
activation1.forward(layer1.output)
layer2.forward(activation1.output)
activation2.forward(layer2.output)
layer3.forward(activation2.output)
activation3.forward(layer3.output)
layer4.forward(activation3.output)
predictions = np.argmax(layer4.output, axis=1)
accuracy = np.mean(predictions == y)

print(f"Accuracy: {accuracy:.4f}")




