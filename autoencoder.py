from keras.datasets import mnist
import numpy as np

# loading mnist data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# flattening the dataset. x_train: 60,000 images, 784 pixels,  x_test: 10,000 images, 784 pixels 
x_train, x_test = x_train.astype('float32') / 255., x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(f'train set:  {x_train.shape}')
print(f'test set:  {x_test.shape}')