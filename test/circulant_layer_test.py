import numpy as np
import tensorflow as tf

from src.layers.structured_layers import Circulant
from src.elm.base_elm import ELM

from scipy.linalg import circulant


def main():
    main_test()


def main_test():
    inputs = [[1, 2, 3], [3, 4, 5]]
    test_circulant_layer(inputs, 2, False)
    test_circulant_layer(inputs, 3, False)
    test_circulant_layer(inputs, 8, False)
    test_circulant_layer(inputs, 9, False, dtype=tf.float16)

    inputs = [[[1, 2], [3, 4]], [[5, 6], [7, 9]]]  # Like tensor
    test_circulant_layer(inputs, 1, False)
    test_circulant_layer(inputs, 2, False)
    test_circulant_layer(inputs, 4, False)
    test_circulant_layer(inputs, 9, False, dtype=tf.float64)

    inputs = [[1, 2, 3], [3, 4, 5]]
    elm = ELM()
    elm.add_dense_layer(10, input_dim=3)
    elm.add_layer(Circulant(35, use_bias=True))
    elm.add_dense_layer(20)
    elm.add_layer(Circulant(3, use_bias=False))
    elm.compile()
    features = elm.get_transformed_features(inputs)
    assert features.shape[0] == 2
    assert features.shape[1] == 3

    print("Circulant test was succesful!")


def test_circulant_layer(inputs, neurons_number, do_printing,
                         dtype=None):
    # process inputs
    inputs = tf.convert_to_tensor(inputs, dtype=dtype or tf.float32)
    inputs_rank = len(inputs.shape)

    # init toeplitz layer
    circulant_layer = Circulant(neurons_number, use_bias=False, activation=None, dtype=dtype)
    circulant_layer.build(inputs.shape)

    # calculate true and our results
    input_len = inputs.shape[-1]
    # Right multiplication interpretation
    dense_matrix = circulant(circulant_layer.get_parameters())
    dense_matrix = tf.transpose(dense_matrix)  # Easier for visualization, so that not shifted weights are in first row.
    dense_matrix = dense_matrix[0:input_len, 0:neurons_number]  # Cut matrix.
    true_result = np.tensordot(inputs, dense_matrix, [[inputs_rank - 1], [0]])[..., 0:neurons_number]
    this_result = circulant_layer.call(inputs)

    if do_printing:
        print(f"x = {inputs}")
        print(f"T = {dense_matrix}")
        print(f"x * T = {true_result}")
        print(f"This layer = {this_result}")

    # Check that results are equal
    difference = true_result - this_result
    tf.assert_greater(tf.convert_to_tensor(0.01, dtype=dtype), difference)

    # Check that output shape is right
    true_output_shape = [dim for dim in inputs.shape]
    true_output_shape[-1] = neurons_number
    tf.assert_equal(this_result.shape, true_output_shape)


if __name__ == '__main__':
    main()
