import numpy as np
from nnl import bundle as nn
import keras.datasets.mnist

#loading the dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

fashion_mnist_labels = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

# preprocessing data
X_test1 = []
X_train1 = []

for i in range(0,len(X_train)):
    X_train1.append(np.reshape(X_train[i],(784)))

for i in range(0,len(X_test)):
    X_test1.append(np.reshape(X_test[i],(784)))

X_train1 = np.array(X_train1)
X_test1 = np.array(X_test1)

# initialize the model
model = nn.Model()

# Add layers
model.add(nn.layer_dense(784, 128))
model.add(nn.relu())
model.add(nn.layer_dense(128, 128))
model.add(nn.relu())
model.add(nn.layer_dense(128, 10))
model.add(nn.softmax())

# Set loss, optimizer and accuracy objects
model.set(
    loss=nn.loss_categoricalCrossentropy(),
    optimizer=nn.optimizer_adam(decay=1e-3),
    accuracy=nn.accuracy_categorical()
)

# Finalize the model
model.finalize()

# Train the model
model.train(X_train1, y_train, validation_data=(X_test1, y_test),
            epochs=10, batch_size=128, print_every=1)

test_input = fashion_mnist_labels[y_test[3]]
predicted = fashion_mnist_labels[np.argmax(model.predict(X_test1[3]))]

print(f'test input image of {test_input}, model predicted {predicted}')