from keras.datasets import mnist
import numpy as np
from tensorflow import keras
from tensorflow.keras import losses
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Dense, InputLayer
import matplotlib.pyplot as plt


# loading mnist data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# flattening the dataset. x_train: 60,000 images, 784 pixels,  x_test: 10,000 images, 784 pixels 
x_train, x_test = x_train.astype('float32') / 255., x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(f'train set:  {x_train.shape}')
print(f'test set:  {x_test.shape}')

# creating the autoencoder model
# input compressed to 64 dim (compression factor of 12.25)

#input_image = Input(shape=(784,))
#encoder = Dense(64, activation='relu')(input_image)
#decoder = Dense(784, activation='relu')(encoder)
#autoencoder = Model(input_image, decoder)

autoencoder = keras.Sequential()
autoencoder.add(InputLayer(input_shape=(784,), name='input'))
autoencoder.add(Dense(64, activation='relu', name='encoder'))   #relu sigmoid
autoencoder.add(Dense(784, activation='sigmoid', name='decoder'))

autoencoder.compile(optimizer='adam', loss='mean_squared_error')
autoencoder.fit(x_train, x_train, epochs=70, validation_split=0.15)

# decode images in the test set 
decoded_images = autoencoder.predict(x_test)

# visualize decoded images to measure performance
plt.figure(figsize=(20, 4))
for i in range(10):
    # original
    plt.subplot(2, 10, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    plt.axis('off')
    # reconstruction
    plt.subplot(2, 10, i + 1 + 10)
    plt.imshow(decoded_images[i].reshape(28, 28))
    plt.gray()
    plt.axis('off')
plt.tight_layout()
plt.show()