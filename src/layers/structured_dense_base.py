from tensorflow.python.framework import dtypes
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.framework import tensor_shape
from abc import ABCMeta, abstractmethod
import numpy as np


class StructuredDense(Dense, metaclass=ABCMeta):  # That means class is abstract.
    """
    Base layer for layers with structured matrices.

    The same as Dense, but matrices are built different way. All subclasses must implement build_structured_kernel.
    All Arguments not listed below are the same as by Dense.

    Arguments:
        units: number of neurones.
        normalize_weights_after_init: if True, weights will be normalized to weight_deviation after initialization.
        weight_deviation: see normalize_weights_after_init.
    """
    def __init__(self,
                 units,
                 normalize_weights_after_init=True,
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
        self.normalize_weights_after_init = normalize_weights_after_init
        self.desired_weight_deviation = weight_deviation
        self.number_of_neurons = units  # Alias for self.units, units is a stupid name.
        self.kernel = None  # Inherited from Dense class. It means weights matrix.
        self.bias = None

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
        self.check_if_weights_are_floating_numbers()
        input_length = get_last_tensor_dimension(input_shape)
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_length})

        self.kernel = self.build_structured_kernel(input_length)
        if self.normalize_weights_after_init:
            self.kernel = self.kernel / np.std(self.kernel) * self.desired_weight_deviation

        if self.use_bias:
            self.bias = self.add_weight(
                'bias',
                shape=[self.units],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True)

        super(Dense, self).build(input_shape)

    def check_if_weights_are_floating_numbers(self):
        dtype = dtypes.as_dtype(self.dtype or K.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError('Unable to build Structure Layer with non-floating point '
                            'dtype %s' % (dtype,))

    @abstractmethod
    def build_structured_kernel(self, input_length):
        """
        Builds structured matrix (aka kernel in tensorflow notation) of the layer.
        Arguments:
            input_length: Length of the inputs.

        Returns:
            kernel: Structured matrix.

        """
        print("Not implemented.")

    def get_parameters(self):
        """
        See returns. The method should be ideally overwritten by subclasses.

        Returns:
            Parameters: Free parameters that are used to build the structured matrix.
        """
        return self.kernel

    def get_dense_matrix(self):
        return self.kernel

    def get_input_length(self):
        return self.input_spec.axes[-1]

    def get_weight_matrix_row(self, row_index):
        return self.kernel[row_index]

    def get_weight_matrix_column(self, column_index):
        return self.kernel[:, column_index]


def get_last_tensor_dimension(shape):
    shape = tensor_shape.TensorShape(shape)
    if tensor_shape.dimension_value(shape[-1]) is None:
        raise ValueError('The last dimension of the inputs Structured Dense Layer'
                         'should be defined. Found `None`.')
    return tensor_shape.dimension_value(shape[-1])
