from keras.datasets import mnist
import numpy as np
from tensorflow import keras
from tensorflow.keras import losses
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation, Dense, InputLayer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# loading mnist data and splitting into training, validation, and testing sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_val = train_test_split(x_train, test_size=0.1, random_state=42)

# flattening the dataset. x_train: 60,000 images, 784 pixels,  x_test: 10,000 images, 784 pixels 
x_train, x_test, x_val = x_train.astype('float32') / 255., x_val.astype('float32') / 255., x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_val = x_val.reshape((len(x_val), np.prod(x_val.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(f'train set:  {x_train.shape}')
print(f'validation set:  {x_val.shape}')
print(f'test set:  {x_test.shape}')

# creating the autoencoder model
# input compressed to 64 dim (compression factor of 12.25)
n_epochs = 20
autoencoder = Sequential()
autoencoder.add(InputLayer(input_shape=(784,), name='input'))
autoencoder.add(Dense(64, activation='relu', name='encoder'))   #relu sigmoid
autoencoder.add(Dense(784, activation='sigmoid', name='decoder'))

autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
history = autoencoder.fit(x=x_train, y=x_train, epochs=n_epochs, validation_data=(x_val, x_val))

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.xticks(np.arange(0, n_epochs, 1))
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('acuracy_epochs.png')
plt.close()

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
plt.savefig('orig_vs_pred.png')
plt.close()