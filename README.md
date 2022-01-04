# Autoencoder that reads hand written digits


## Blog 
My blog on this project can be accessed at [TO BE POSTED]

## Dataset
https://github.com/davidflanagan/notMNIST-to-MNIST

## Autoencoder 
Neural Networks can be used for supervised and unsupervised learning tasks. In supervised learning methods, Neural Networks are trained using
labeled training dataset. An autoencoder neural network is an unsupervised learning method in which the target values are set to be equal to 
the inputs. In other words, the network uses y_i = x_i. One specific property of an Autoencoder is that a "bottleneck" is imposed in the hidden
layer such that a compressed knowledge representation of the input is learned and the output is simply a reconstruction of the input. An important
thing to note here is that the compression and reconstruction task is extremely difficult if the inputs are independent of each other or random. 
Autoencoders work well with input features that are correlated. The following visual shows example of an Autoencoder.
![Autoencoder](https://github.com/arqchicago/autoencoder/autoencoder.png)



