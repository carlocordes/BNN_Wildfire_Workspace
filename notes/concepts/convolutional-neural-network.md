# Convolutional Neural Network

Similar to ordinary [[neural-network]], only the explicit assumption is made that the inputs are images. This assumption allows for certain extra information to be encoded in the network architecture, drastically reducing the amounts of network parameters needed.

Image size dictates the applicability of regular architectures. An RGB image of size 3x32x32 would impose 3070 weights for every neuron in the first hidden layer. This does not scale well to larger images.

CNN's arrange neurons in 3, rather than 2 dimensions: **width, depth & height**.

The three main layers used in convNet are:
1. **Convolutional Layer:** Performs the convolution operation, extracting features from the input image.
    - A [[kernel]] (e.g. $M_{3X3}$)contains learneable weights which are to be trained. Every kernel is designed to detect one feature. It is convolved with the image, reducing its dimensions accordingly
    - Output neurons in convolutional layers are not connected to the entire image, but specific regions as defined by the kernel
    - The concept hinges on the fact, that images are not just composed by pixels, but distinct features (edges, corners, curves etc.), which the network should learn. This is refered to as [[spatial-hierarchy]]. By applying convolutions through kernels, different locations in the input having similar properties "share weights" in the kernels matrix

2. **Pooling Layer**
    - [[pooling]] relates to the idea of translation invariance in an input image
    - Layer is integrated right after the activation function of a convolutional layer
    - Down-sampling of feature map
    - Matrix slides accross image producing new values and reducing amount of values
3. **Classification Head (FC Layer)**
    - Consists of one or more fully-connected layers as in a standard feedforward neural networ
    - Pooling layer outputs a 3D volume, the classification head requires a 1D array, which is achieved by [[flattening]]
    - This layer converts the extracted features into final classification scores

