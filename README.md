# THIS PAGE NOT READY YET

# My-TF-AutoEncoder
Basic and Generic AutoEncoder for Multifunctional purposes

## Overview
AutoEncoders are **unsupervised Learning** tools that have many uses in **machine learning**. It can be used for many purposes such as: **dimension reduction**, **information retrieval**, **anomaly detection** and **noise filtering**. This code is a tensorflow-based implemnatation of a **Convolutional Neural Network AutoEncoder**. The code allows a simple definition of the hyperparameters of the network and easy way to build & train the AutEncoder network.

![simple_prediction](https://github.com/EtzionR/My-TF-AutoEncoder/blob/main/outputs/predict.png)

(All examples imported for this code are based on [MNIST Dataset](http://yann.lecun.com/exdb/mnist/))

The AutoEncoder is build from two main parts: the **Encoder** and the **Decoder**. Each of them composed of several layers, as can be seen in the following example:

![network_draw](https://github.com/EtzionR/My-TF-AutoEncoder/blob/main/outputs/network.png)

(You can see that in this case, the number of convolution layers in each part is 2 and so is the number of filters)

### Encoding
The Encoder is composed of layers. The basic layer is the **convolution layer**, which is defined by the amount of filters (which increase exponentially as the network deepens) and the size of their kernel. In addition, this layer has a **relu activation** function. Each time the layer uses weights matrix as a mapping filter, and multiplies the matrix of the filter with the selected area from the original matrix so that at the end one value is calculated for the target cell in the output matrix. Because we defined serval filters, the output of such layer is a **3D Tensor**, and not a simple matrix. 

Another layer (which the code Automatically adds after a convolution layer) is the **maxpooling layer**. This layer reduces the dimensions of the matrix by half. It do that by take each 2X2 square of cells in the input matrix and keep only the highest value from the square.

After several layers of convolution and pooling, we want to convert our tensor to a vector in some small dimension (relatively). To do this, we perform a **flattening** and multiplication operation in the matrix (standard **dense layer**). The vector we obtained exists in a **latent space** and use as the **representative vector** of the object we input to the Encoder. We will use this vector in the decoding process.

### Decoding
The decoding process also consists several layers of t_conv (**transpose convulotion**) that reveresed the original convolution process. Each cell in the input tensor multiplies by a specific map filter. so it converted into a sub-matrix with kernel size, in the output matrix.

In addition, after each t_conv layer, the network also includes an **upsampling** layer that reverses the process performed by the pooling layers. These layers return for each individual cell a 2X2 square of cells with equal value to the signle original value in the input tensor.

In this way, the network tries to match the representing vector to desired output. At the end of the process, we test our output against the actual required output, which is defined as our Y. By this process, the network "learn" and adapt the layers weights. We update the weights of the network using **adam optimizer** when our loss function is **MSE**. This is how the network actually learns along the various stages to adapt itself to the required output. You can see in the following example how the AutoEncoder tries over 150 epochs to restore the original shape of the character "5" inserted into it:

As you can see, at first the network is not at all close to creating something resembling the desired character, but within a few steps, it already manages to create an image reminiscent of the desired character, until at the end of the process there is almost no noise output and the story can be clearly seen.

As mentioned, automatic coders can be used for many purposes, we will bring here some basic examples:
As mentioned, coders can be used to reproduce the same matrix they receive. That is, you feed the same matrix as both input (X) and output (Y) of the network. In this case, the weights of the network will learn how to recover from the vector representing the original matrix, as can be seen in the following example:

Also, after network training, the representing vector can be used as a dimensional reduction of the original matrix (similar to PCA). Vector exists in a latent space, and it can be seen that in the case before us, different numbers are given a different latent representation:

(This example presents of course a case where it is chosen that the length of the representing vector is 2, so that it can be used as two-dimensional coordinates).
The network can also be used to filter noise in the given data. To do this, we will tune the network so that its input will be synthetically inserted noise matrices, while the output will be noise-free matrices. Now, if we try to insert a noisy matrix into the network, the network will correct the noises and return a "clean" matrix, as can be seen in the following example:

Similarly, a network can be used to detect anomalies in our dataset. In the following example, the network is fostered only on standard, noiseless matrices. After training, we used the net to return a prediction on a new dataset. In this dataset, we entered a minimal amount (3%) of observations with synthetic noises. The network, of course, fostered a regular and clean database, which of course led to a prediction that was far from the entered data. Therefore, by measuring the distances, only the matrices that constitute anomalies can be isolated from each infected date, as can be seen in the following example:

As you can see, the matrices we isolated here are indeed full of noise and form part of the infected observations we added synthetically - which means that we were able to identify the anomalies from the whole data set and isolate them.

## Libraries
The code uses the following library in Python:

**matplotlib**

**numpy**

## Application
An application of the code is attached to this page under the name: 

[**implementation.pdf**]()

The examples are also attached here [Data]().


## Example for using the code
To use this code, you just need to import it as follows:
``` sh
# import code

# load data

# define variables

# using the code

```

When the variables displayed are:

**v:** hh

**b:** hh (defualt = 5)

## License
MIT © [Etzion Harari](https://github.com/EtzionR)

