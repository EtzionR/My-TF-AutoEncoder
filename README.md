# My-TF-AutoEncoder
Basic and Generic AutoEncoder for Multifunctional purposes

## Overview
AutoEncoders are **unsupervised Learning** tools that have many uses in **machine learning**. It can be used for many purposes such as: **dimension reduction**, **information retrieval**, **anomaly detection** and **noise filtering**. This code is a TensorFlow-based implementation of a **Convolutional Neural Network AutoEncoder**. The code allows a simple definition of the hyperparameters of the network and easy way to build & train the AutEncoder network.

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

In this way, the network tries to match the representing vector to desired output. At the end of the process, we test our output against the actual required output, which is defined as our Y. By this process, the network "learn" and adapt the layers weights. We update the weights of the network using **adam optimizer** when our loss function is **MSE**. This is how the network actually learns along the various stages to adapt itself to the required output. You can see in the following example how the AutoEncoder learn over 150 epochs to restore the original shape of the character "5" inserted into it:

![learning](https://github.com/EtzionR/My-TF-AutoEncoder/blob/main/outputs/learning_fig.gif)

As you can see, at first, the network is not close at all to creating something resembling the desired character, but within a few steps, it already manages to create an image close of the desired character, until at the end of the process there is almost no noise in the output and the character can be clearly seen.

AutoEncoder can be used for many applications, we will bring here some basic examples:
- As mentioned, coders can be used to **retrive** the same matrix they receive. To do so, we feed the same matrix as both input (X) and output (Y) to the network. In this case, the weights of the network will learn how to recover the original matrix from the representing vector (as also to encoded the input matrix to this vector). You can see such process in the following example:

![retrive](https://github.com/EtzionR/My-TF-AutoEncoder/blob/main/outputs/encode_decode.gif)

- Also, after network training, the representing vector can be used as a **dimensional reduction** of the original matrix (similar to **PCA**). The vector coordiantes exists in a latent space, and it can be seen that in the case before us, different numbers are given a different latent coordinate representation:

![latent_space](https://github.com/EtzionR/My-TF-AutoEncoder/blob/main/outputs/latant_2d.png)

&emsp;&emsp; (This example presents of course a case where it is chosen that the length of the representing vector is 2, so that it can be used as 2D coordinates).

- The network can also be used to **filter noise** in the given data. To do this, we will train the network so that its input will be synthetically inserted noise matrices, while the output will be noise-free matrices. Now, if we try to insert a noisy matrix into the network, the network weights will filter the noises and return a "clean" matrix, as can be seen in the following example:

![noise_filter](https://github.com/EtzionR/My-TF-AutoEncoder/blob/main/outputs/noisy.gif)

- Similarly, the AutoEncoder can be used to **detect anomalies** in our dataset. In the following example, the network is train only on standard, noiseless matrices. After training, we used the network to return a prediction on a new dataset. In this dataset, we entered a minimal amount (**3%**) of matrices with **synthetic noises**. The network, of course, trained based on regular and clean database, which of course led to a prediction that was far from the entered data. Therefore, by measuring the distances between the actual matrix to the prediction we can detected unusual matrices. By doing so, we can **isolated the anomalies** in the given dataset, as can be seen in the following example:

![anomaly](https://github.com/EtzionR/My-TF-AutoEncoder/blob/main/outputs/anomaly.png)

&emsp;&emsp; As you can see, the matrices we isolated here are indeed full of noise and was part of the infected matrices we added synthetically - which means that we were able to identify the anomalies from the whole data set and isolate them.

After you training the model, you might want to see the **learning curve** progress by epochs. To do so, you can use the built-in plot function of the object, as the following example:
``` sh
# import code
from cec import AutoEncoder

# fitting network
autoencoder = AutoEncoder(source=(28,28),kernels=[5,5],filters=3,latant_dim=100,epochs=150,lr=.001).fit(input_x,output_y)

# plot loss
autoencoder.plot_loss()
```
![loss_curve](https://github.com/EtzionR/My-TF-AutoEncoder/blob/main/outputs/loss.png)

It's also possible to save the model weights. so you can build & train the model, and just load it to new **AutoEncoder** object and use in the trained model:

``` sh
# import code
from cec import AutoEncoder

# fitting network
autoencoder = AutoEncoder(source=(28,28),kernels=[5,5],filters=3,latant_dim=100,epochs=150,lr=.001).fit(input_x,output_y)

# save model
autoencoder.save(path = 'model')

# load model
autoencoder_2 = AutoEncoder(...).load('model')

# prediction (by the trained weights)
autoencoder_2.predict(input_data)
```

For conculsion, we saw that AutoEncoders as a lot of useful applications, that can be serve as useful tools in the field of machine learning.

## Libraries
The code uses the following library in Python:

**tensorflow**

**matplotlib**

**numpy**

## Application
An application of the code is attached to this page under the name: 

[**implementation.pdf**](https://github.com/EtzionR/My-TF-AutoEncoder/blob/main/implementation.pdf)

The examples are also attached here [data](https://github.com/EtzionR/My-TF-AutoEncoder/tree/main/data).


## Example for using the code
To use this code, you just need to import it as follows:
``` sh
# import code
from cec import AutoEncoder
import numpy as np

# load data
x = np.load('x.npy') # training input
y = np.load('y.npy') # training output
z = np.load('z.npy') # new data for prediction

# define variables
source = (28,28)
kernels = [5,5]
filters = 3
latant_dim = 100
epochs = 150
lr = 0.001

# using the code
aec = AutoEncoder(source=source,kernels=kernels,filters=filters,latant_dim=latant_dim,epochs=epochs,lr=lr)

# fitting the model
aec.fit(x,y)

# get prediction
aec.predict(z)
```

When the variables displayed are:

**source:** tuple of ints, the input matrix source shape

**kernels:** list of ints, the size of the kernel in each layer

**filters:** int, number of filters to add in each layer

**latant_dim:** int, dimension for the latatnt representation code(defualt = 2)

**epochs:** int, number of training epochs (defualt = 150)

**lr:** float, the learning rate  (defualt = .001)

## License
MIT ?? [Etzion Harari](https://github.com/EtzionR)

