# Convolutional Neural Network

Similar to ordinary [[neural-network]], only the explicit assumption is made that the inputs are images. This assumption allows for certain extra information to be encoded in the network architecture, drastically reducing the amounts of network parameters needed.

Image size dictates the applicability of regular architectures. An RGB image of size 3x32x32 would impose 3070 weights for every neuron in the first hidden layer. This does not scale well to larger images.

CNN's arrange neurons in 3, rather than 2 dimensions: **width, depth & height**.

