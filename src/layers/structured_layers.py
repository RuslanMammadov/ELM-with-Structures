import tensorflow as tf
import numpy as np

import scipy
from scipy.linalg import hadamard

from src.layers.structured_dense_base import StructuredDense


class Circulant(StructuredDense):
    """ Layer with Circulant weight matrix. """

    def build_structured_kernel(self, input_length):
        params_number = max(self.number_of_neurons, input_length)

        params = self.add_weight(
            'kernel',
            shape=[params_number],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True)

        # For better visualization: Matrix T = [row 1, row 2, ...]
        # Right multiplication interpretation, neurons = input * T.
        # So n columns => n number of neurons, m rows => m = input length
        repeated_two_times_params = tf.tile(params, [2])  # [*params, *params]
        rows = []
        for i in range(input_length):
            start_index = params_number - i
            row = repeated_two_times_params[start_index: start_index + self.number_of_neurons]
            rows.append(row)
        return tf.stack(rows, axis=0)

    # you have to draw the circulant matrix to understand what is going here
    def get_parameters(self):
        if self.number_of_neurons >= self.get_input_length():  # Row is bigger then column
            return self.get_weight_matrix_row(0)
        else:
            first_column = self.get_weight_matrix_column(0)
            parameters = tf.reverse(first_column[1:], axis=[0])
            parameters = tf.concat([first_column[0:1], parameters], axis=0)  # prepend first value
            return parameters


class Toeplitz(StructuredDense):
    """ Layer with Toeplitz weight matrix. """

    def build_structured_kernel(self, input_length):
        params = self.add_weight(
            'kernel',
            shape=[input_length + self.number_of_neurons - 1],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True)

        # For better visualization: Matrix T = [row 1, row 2, ...]
        # Right multiplication interpretation, units = input * T.
        # So n columns => n units, m rows => m = input dim
        rows = []
        for i in range(input_length):
            start_index = input_length - i - 1
            row = params[start_index: start_index + self.number_of_neurons]
            rows.append(row)
        return tf.stack(rows, axis=0)


class LowRank(StructuredDense):
    """ Layer with weight Matrix that has low rank (rank smaller than both dimensions). """

    def __init__(self,
                 units,
                 rank,
                 normalize_weights_after_init=True,
                 weight_deviation=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        """
        Arguments not listed below are the same as by the superclass StructuredDense.

        Arguments:
            rank: Rank of the matrix.
        """
        self.rank = rank
        super().__init__(
            units,
            normalize_weights_after_init=normalize_weights_after_init,
            weight_deviation=weight_deviation,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)

    def build_structured_kernel(self, input_dim):
        matrix_1 = self.add_weight(
            'kernel',
            shape=[input_dim, self.rank],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True)

        matrix_2 = self.add_weight(
            'kernel',
            shape=[self.rank, self.units],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True)

        return tf.matmul(matrix_1, matrix_2)


class Vandermonde(StructuredDense):
    """
    Vandermonde Layer.
    Multiplication matrix:
    [1, a, a^2, ... a^n
     1, b, b^2, ... b^n]
     """

    def __init__(self,
                 units,
                 normalize_weights_after_init=True,
                 weight_deviation=1,
                 drop_ones=False,
                 transpose=False,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        """
        Arguments not listed below are the same as by superclass.

        Arguments:
            drop_ones: if True, the column or row with only ones will be dropped.
            transpose: if True, the weights for the same neurons will have same power, for example for 3th neuron
                weights will be a^2, b^2, ... or a^3, b^3 ... if drop_ones=True.
                        If False, the weights for the same inputs will have same power, for example for 3th input feature
                weights will be a^2, b^2, ... or a^3, b^3, ... if drop_ones=True.
        """

        self.drop_ones = drop_ones
        self.transpose = transpose
        super().__init__(
            units,
            normalize_weights_after_init=normalize_weights_after_init,
            weight_deviation=weight_deviation,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)

    def build_structured_kernel(self, input_length):
        number_of_parameters = self.number_of_neurons if not self.transpose else input_length
        params = self.add_weight(
            'kernel',
            shape=[number_of_parameters],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True)

        # For better visualization: Matrix T = [row 1, row 2, ...]
        # Right multiplication interpretation, units = input * T.
        # So n columns => n units, m rows => m = input dim.
        if not self.transpose:
            rows = []
            for i in range(input_length):
                n = i + 1 if self.drop_ones else i
                row = tf.pow(params, n)
                rows.append(row)
            return tf.stack(rows, axis=0)
        else:
            columns = []
            for i in range(self.number_of_neurons):
                n = i + 1 if self.drop_ones else i
                column = tf.pow(params, n)
                columns.append(column)
            return tf.stack(columns, axis=1)

    def get_parameters(self):
        index = 1 if not self.drop_ones else 0
        return self.get_weight_matrix_row(index) if not self.transpose else self.get_weight_matrix_column(index)


class FastFood(StructuredDense):
    """
    Fastfood Transformation Matrix Layer: V = SHGPHB
    Binary scaling matrix B is a diagonal matrix with -1 and 1 values.
    H is the Wash-Hadamard matrix.
    P is permutation matrix.
    G is Gaussian diagonal scaling matrix which entries are generated by N(0, 1).
    See: Q. Le, T. Sarlos, and A. Smola. “Fastfood: Approximate Kernel Expansions in LoglinearTime”
    """

    def build_structured_kernel(self, input_length):
        # This method is effective if output is bigger than input.
        # In other case more effective method should be programmed.

        # Single Fastfood matrix will be d X d
        self.d = input_length if is_power_of_two(input_length) else get_next_power_of_two(input_length)
        self.number_of_fastfoods = self.number_of_neurons // self.d + int(self.number_of_neurons % self.d > 0)

        P = self.build_P_matrix()
        B = self.build_B_matrix()

        self.G = self.add_weight(
            'kernel',
            shape=[self.number_of_fastfoods * self.d],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype)

        return self.create_kernel_matrix(input_length, B, P)

    def build_P_matrix(self):
        permutation_list = []
        for i in range(self.number_of_fastfoods):
            start_number = i * self.d
            permutation_numbers = range(start_number, start_number + self.d)
            permutation_list.append(np.random.permutation(permutation_numbers))
        permutation_list = np.concatenate(permutation_list, axis=0)
        P = create_permutation_matrix(permutation_list)
        P = tf.cast(P, dtype=self.dtype)
        return P

    def build_B_matrix(self):
        B = [np.diag(np.random.choice([-1., 1.], self.d)) for i in range(self.number_of_fastfoods)]
        B = np.concatenate(B, axis=1)
        B = tf.cast(B, dtype=self.dtype)
        return B

    def create_kernel_matrix(self, input_dim, B, P):
        hadamard_matrix = scipy.linalg.hadamard(self.d)
        size_before_cutting = self.d * self.number_of_fastfoods
        
        H = np.zeros([size_before_cutting, size_before_cutting])
        for i in range(self.number_of_fastfoods):
            start_i = i * self.d
            H[start_i:start_i + self.d, start_i:start_i + self.d] = hadamard_matrix
        H = tf.cast(H, dtype=self.dtype)

        G = tf.linalg.diag(self.G)
        if not hasattr(self, "test"):
            self.G = None

        kernel = tf.matmul(B, H)
        kernel = tf.matmul(kernel, P)
        kernel = tf.matmul(kernel, G)
        kernel = tf.matmul(kernel, H)
        kernel = kernel[:input_dim, :self.units]
        kernel = kernel / (self.d ** 0.5)
        return kernel


def is_power_of_two(n):
    return (n != 0) and (n & (n - 1) == 0)  # stack overflow!


def get_next_power_of_two(n):
    return int(2 ** (np.floor(np.log2(n)) + 1))


def create_permutation_matrix(permutation_list):
    length = len(permutation_list)
    perm_matrix = np.zeros([length, length])
    for i in range(length):
        perm_matrix[i, permutation_list[i]] = 1
    return perm_matrix


class ToeplitzLike(StructuredDense):
    """
    Toeplitz Like Structured Matrix.

    M(G, H) = 􏰊 sum over i from 0 to rank: Z_1(gi)Z_minus_1(hi).
    Z_1 - circulant matrix and Z_minus_1 is skewed circulant matrix.
    See: Vikas Sindhwani, Tara N. Sainath, Sanjiv Kumar: Structured Transforms for Small-Footprint Deep Learning
    """

    def __init__(self,
                 units,
                 rank=2,
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
        """
        Arguments not listed below are the same as by the superclass.

        Important:
            If normalize_weights_after_init=False, the weights can have unexpected too big or small deviation.

        Arguments:
            rank: Low displacement rank of ToeplitzLike Structured matrix.
            normalize_weights_after_init: normalize toeplitz like matrix so that deviation of weight are equal dev.
                Important: kernel_initializer must generate centered numbers if normalize=True.
                It is recommended to use it with gaussian(normal) initializer.
            weight_deviation: if normalize is True, weight will have deviation=dev
        """
        self.rank = rank

        super().__init__(
            units,
            normalize_weights_after_init=normalize_weights_after_init,  # we do not normalize here anymore (in this class/abstract level)
            weight_deviation=weight_deviation,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)

    def build_structured_kernel(self, input_length):
        # This method is effective if output is bigger than input.
        # In other case more effective method should be programmed.

        # Single ToeplitzLike matrix will be n X n
        n = max(input_length, self.number_of_neurons)

        G = self.add_weight(
            'kernel',
            shape=[self.rank, n],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype)

        H = self.add_weight(
            'kernel',
            shape=[self.rank, n],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype
        )

        return get_toeplitz_like_matrix(G, H)[:input_length, :self.number_of_neurons]


def get_toeplitz_like_matrix(G, H_reversed):
    assert G.shape == H_reversed.shape
    (rank, n) = G.shape  # Single ToeplitzLike matrix will be n X n

    toeplitz_like_matrix = tf.zeros([n, n])
    for i in range(rank):
        Z_g = get_circulant_matrix(G[i, :])
        Z_skew_h = get_circulant_matrix(H_reversed[i, :], is_skew=True)
        toeplitz_like_matrix += tf.matmul(Z_g, Z_skew_h)
    return toeplitz_like_matrix * 0.5


def get_circulant_matrix(params, is_skew=False):
    number_of_params = len(params)
    if is_skew:
        expanded_params = tf.concat([tf.scalar_mul(-1, params), params], axis=0)
    else:
        expanded_params = tf.tile(params, [2])
    columns = []
    for i in range(number_of_params):
        start_index = number_of_params - i
        column = expanded_params[start_index: start_index + number_of_params]
        columns.append(column)
    return tf.stack(columns, axis=1)
