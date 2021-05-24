import torch
from torch import nn
import tensorflow as tf
import keras
import keras.layers as KL

# init_value = 1.0
# x = torch.FloatTensor([init_value])

# print(x.shape)


class _Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(_Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class Scale(tf.keras.layers.Layer):
    def __init__(self):
        super(Scale, self).__init__()

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape,
            dtype='float32',
            initializer='ones',
            trainable=True
        )

    def call(self, inputs):
        return inputs * self.scale


if __name__ == "__main__":
    _input = torch.rand(4, 3, 256, 256)
    input = tf.random.normal((4, 256, 256, 3))
    print(_input.shape)
    cacu = _Scale(init_value=1.0)
    _x = cacu(_input)
    print(_x.shape)
    # cacu = Scale()
    # x = cacu(input)
    # print(x.shape)

    _x = torch.exp(_x)
    print(_x.shape)
