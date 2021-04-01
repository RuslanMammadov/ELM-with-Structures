import numpy as np
import tensorflow as tf

from src.layers.structured_layers import Vandermonde
from src.elm.base_elm import ELM


def main():
    main_test()


def main_test():
    inputs = [[1, 2, 3], [3, 4, 5]]
    test_vandermonde_layer(inputs, 2, False)
    test_vandermonde_layer(inputs, 3, False)
    test_vandermonde_layer(inputs, 8, False)
    test_vandermonde_layer(inputs, 9, False, dtype=tf.float16)

    inputs = [[[1, 2], [3, 4]], [[5, 6], [7, 9]]]  # Like tensor
    test_vandermonde_layer(inputs, 1, False)
    test_vandermonde_layer(inputs, 2, False)
    test_vandermonde_layer(inputs, 4, False)
    test_vandermonde_layer(inputs, 9, False, dtype=tf.float64)

    test_elm_with_vandermonde_layers()

    print("Vandermonde test was succesful!")


def test_elm_with_vandermonde_layers():
    inputs = [[1, 2, 3], [3, 4, 5]]
    elm = ELM()
    elm.add_dense_layer(10, input_dim=3)
    elm.add_layer(Vandermonde(35, use_bias=True))
    elm.add_dense_layer(20)
    elm.add_layer(Vandermonde(3, use_bias=False, drop_ones=True))
    elm.compile()
    features = elm.get_transformed_features(inputs)
    assert features.shape[0] == 2
    assert features.shape[1] == 3


def test_vandermonde_layer(inputs, neurons_number, do_printing,
                           dtype=None):
    # process inputs
    inputs = tf.convert_to_tensor(inputs, dtype=dtype or tf.float32)
    inputs_rank = len(inputs.shape)

    # init toeplitz layer
    vandermonde_layer = Vandermonde(neurons_number, normalize_weights_after_init=False, use_bias=False, activation=None,
                                    dtype=dtype)
    vandermonde_layer.build(inputs.shape)

    # calculate true and our results
    dense_matrix = np.vander(vandermonde_layer.get_parameters(), vandermonde_layer.get_input_length(), increasing=True)
    dense_matrix = np.transpose(dense_matrix)  # For right multiplication
    true_result = np.tensordot(inputs, dense_matrix, [[inputs_rank - 1], [0]])
    this_result = vandermonde_layer.call(inputs)

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
