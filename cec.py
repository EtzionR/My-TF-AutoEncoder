# Create by Etzion Harari
# https://github.com/EtzionR

# Load libraries:
from tensorflow.keras import layers, losses, Sequential, optimizers
from tensorflow.keras.models import Model, load_model
import matplotlib.pyplot as plt
import numpy as np

# Define useful functions:
adam     = lambda lr: optimizers.Adam(learning_rate=lr)

# Global Variables
LOSS     = 'loss'
RELU     = 'relu'
POOL     = 2

# TensorFlow Layers & losses
MSE      = losses.MeanSquaredError()
source   = layers.Input
conv     = layers.Conv2D
tconv    = layers.Conv2DTranspose
reshape  = layers.Reshape
dense    = layers.Dense
flatten  = layers.Flatten
maxpool  = layers.MaxPooling2D
upsample = layers.UpSampling2D

# Define AutoEncoder Object:
class AutoEncoder:
    """
    CNN AutoEncoder
    Build Deep Convolutional AutoEncoder

    The user can defined the depth of the Network,
    The code latant dimension, learning rate, number of filters
    and kernels size
    """
    def __init__(self, source, kernels, filters, latant_dim=2, epochs=100, lr=1e-3):
        """
        initilize the AutoEncoder Object
        :param source: tuple of two ints, the shape of given input
        :param kernels: list of D ints, the kernel sizes (when D referred to Network depth)
        :param filters: int, number of filters to product each layer
        :param latant_dim: int, the encoded latant dimension (default: 2)
        :param epochs: int, number of epoch for training (default: 100)
        :param lr: float, learning rate (default: .001)
        """
        self.latant_dim = latant_dim
        self.kernels = [(k, k) for k in kernels]
        self.filters = filters
        self.source = (*source,) + (1,)
        self.epochs = epochs
        self.lr = lr

        self.loss = []
        self.model = None
        self.reshp = self.reshape_dim()
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    def reshape_dim(self):
        """
        calculate the dimension of the last encoded layer
        :return: last layer shape
        """
        shape = self.source
        for k, _ in self.kernels:
            klen = k - 1
            shape = (int((shape[0]-klen)/POOL), int((shape[1]-klen)/POOL), shape[2]*self.filters)
        return shape

    def build_encoder(self):
        """
        build Deep Convolutional Encoder
        In each step we add also conv layer and maxpooling layer
        In the Last step, we also add flatten & dense layer to
        convert the tensor into single representation vector
        :return: Encoder (with untrained weights)
        """
        layers = []
        depth = 1
        for kernel in self.kernels:
            depth *= self.filters
            layers.append(conv(depth, kernel, activation=RELU))
            layers.append(maxpool())
        layers += [flatten(), dense(self.latant_dim, activation=RELU)]
        return Sequential(layers)

    def build_decoder(self):
        """
        build Deep Convolutional Decoder
        In each step we add also transope conv layer and unsampling layer
        in the first step we add dense & reshape layers to convert
        the representation vector into tensor
        :return: Decoder (with untrained weights)
        """
        layers = [source(shape=(1, self.latant_dim)),
                  dense(np.prod(self.reshp), activation=RELU),
                  reshape(target_shape=self.reshp)]
        depth = self.reshp[-1]
        for kernel in self.kernels[::-1]:
            depth /= self.filters
            layers.append(upsample())
            layers.append(tconv(depth, kernel, activation=RELU))
        return Sequential(layers)

    def fit(self, x, y):
        """
        fitting the data through encoding-decoding proccess
        :param x: input matrices to encoding
        :param y: requied output from the network
        """
        start = source(shape=self.source)
        model = Model(inputs=start, outputs=self.decoder(self.encoder(start)))
        model.compile(optimizer=adam(self.lr), loss=MSE)
        model.fit(x, y, epochs=self.epochs, shuffle=True)
        self.model = model
        self.loss += model.history.history[LOSS]
        return self

    def predict(self, x):
        """
        predict the outputs of the network from given x
        :param x: given input for prediction
        :return: prediction
        """
        return self.model.predict(x)

    def encode(self, x):
        """
        encoding input x
        :param x: input x for encoding
        :return: encoded x in latant dimension
        """
        return self.encoder(x)

    def decode(self, v):
        """
        decoding input vector
        :param v: input vector for decoding from latant dimension
        :return: decoded matrix
        """
        return self.decoder(v)

    def save(self,path):
        """
        save the model parts as h5 files
        :param path: given file path to the saved model
        """
        self.encoder.save(f'{path}\Encoder.h5')
        self.decoder.save(f'{path}\Decoder.h5')
        self.model.save(f'{path}\Autoencoder.h5')
        return self

    def load(self,path):
        """
        load model parts h5 files from given path
        :param path: given file path to the saved model
        """
        self.encoder = load_model(f'{path}\Encoder.h5')
        self.decoder = load_model(f'{path}\Decoder.h5')
        self.model   = load_model(f'{path}\Autoencoder.h5')
        return self

    def plot_loss(self,size=10):
        """
        plot the loss by the epochs
        """
        plt.figure(figsize=(size, size*.6))
        plt.title(f'Loss Value for AutoEncoder\nepoch number = {self.epochs}', fontsize=16)
        plt.plot(np.arange(len(self.loss)), self.loss, color='r', label=LOSS)
        plt.xlabel('number of epoch', fontsize=14)
        plt.ylabel('loss value', fontsize=14)
        plt.ylim(0, self.loss[0] * 1.05)
        plt.show()

# License
# MIT © Etzion Harari