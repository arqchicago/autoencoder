from keras.datasets import mnist
import numpy as np
from tensorflow import keras
from tensorflow.keras import losses
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Dense

# loading mnist data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# flattening the dataset. x_train: 60,000 images, 784 pixels,  x_test: 10,000 images, 784 pixels 
x_train, x_test = x_train.astype('float32') / 255., x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(f'train set:  {x_train.shape}')
print(f'test set:  {x_test.shape}')


# creating the autoencoder model
# input compressed to 32 dim (compression factor of 24.5)
model = keras.Sequential()
model.add(keras.Input(shape=(784,), name='layer1'))
model.add(Dense(32, activation='relu', name='layer2'))   #relu sigmoid
model.add(Dense(784, activation='sigmoid', name='layer3'))

print(model.summary())
print(f'input shape= {model.input_shape}')
print(f'output shape= {model.output_shape}')

model.compile(optimizer='adam', loss=losses.MeanSquaredError())
model.fit(x_train, x_train, epochs=75, validation_split=0.10, verbose=1)