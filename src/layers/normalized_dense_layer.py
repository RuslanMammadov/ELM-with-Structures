from tensorflow.python.keras.layers import Dense
import tensorflow as tf
import numpy as np


class NormalizedDense(Dense):
    """
    Normalized densely-connected NN layer.

    After dense layer is created, weights will be normalized so that the deviation is equal weight_deviation.

    Arguments that are not listed below are the same as by Dense.

    Arguments:
        weight_deviation: which deviation should weights have after creation.
    """

    def __init__(self,
                 units,
                 weight_deviation=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_normal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        self.weight_deviation = weight_deviation
        super().__init__(
            units=units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )

    def build(self, input_shape):
        super().build(input_shape)

        np_kernel = np.asarray(self.kernel.numpy(), dtype=self.dtype or tf.float32)
        real_deviation = np.std(np_kernel)
        self.kernel.assign(self.kernel / real_deviation * self.weight_deviation)
