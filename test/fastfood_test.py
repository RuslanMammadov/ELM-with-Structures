import numpy as np
import tensorflow as tf

from src.layers.structured_layers import FastFood
from src.elm.base_elm import ELM
from matplotlib import pyplot as plt

from tensorflow.keras.initializers import RandomUniform, RandomNormal

import os

os.environ['KMP_DUPLICATE_LIB_OK'] ='True'
# from src.elm.base_elm import ELM


def main():
    main_test()


def main_test():
    inputs = [[1, 2, 3], [3, 4, 5]]
    test_fastfood_layer(inputs, 2)
    test_fastfood_layer(inputs, 3)
    test_fastfood_layer(inputs, 8)
    test_fastfood_layer(inputs, 9, dtype=tf.float16)

    inputs = [[[1, 2], [3, 4]], [[5, 6], [7, 9]]]  # Like tensor
    test_fastfood_layer(inputs, 1)
    test_fastfood_layer(inputs, 2)
    test_fastfood_layer(inputs, 4)
    test_fastfood_layer(inputs, 9, dtype=tf.float64)

    test_elm_with_fastfood_layer()

    print("Fastfood test was succesful!")


def test_elm_with_fastfood_layer():
    inputs = [[1, 2, 3], [3, 4, 5]]
    elm = ELM()
    elm.add_dense_layer(10, input_dim=3)
    elm.add_layer(FastFood(35, use_bias=True))
    elm.add_dense_layer(20)
    elm.add_layer(FastFood(3, use_bias=False))
    elm.compile()
    features = elm.get_transformed_features(inputs)
    assert features.shape[0] == 2
    assert features.shape[1] == 3


def test_fastfood_layer(inputs, neurons_number, dtype=None):
    # process inputs
    inputs = tf.convert_to_tensor(inputs, dtype=dtype or tf.float32)

    # init toeplitz layer
    fastfood = FastFood(neurons_number, use_bias=False, activation=None, dtype=dtype)
    fastfood.build(inputs.shape)

    # calculate true and our results
    result = fastfood.call(inputs)

    # Check that output shape is right
    true_output_shape = [dim for dim in inputs.shape]
    true_output_shape[-1] = neurons_number
    tf.assert_equal(result.shape, true_output_shape)


def interesting_weights_plot():
    plot_weights(RandomUniform(-1, 1), "Uniform")
    plot_weights(RandomNormal(-2, 1), "Normal. Mean=-2")
    plot_weights(RandomNormal(0, 1), "Normal.")
    plt.close("all")


def plot_weights(distribution, name=""):
    fastfood_layer = FastFood(128, use_bias=False, activation=None, kernel_initializer=distribution)
    fastfood_layer.test = True
    fastfood_layer.build([1, 128])

    kernel = np.asarray(fastfood_layer.kernel)
    full_kernel = kernel.flatten()
    first_row = kernel[0, :]
    second_row = kernel[1, :]
    G = fastfood_layer.G.numpy()
    G = np.flatten(G)

    fig = plt.figure()
    if name != "":
        fig.suptitle(name)

    plt.subplot(3, 1, 1)
    plt.hist(full_kernel)
    plt.title(f"Weights: Mean:{np.mean(full_kernel):.{2}}, Var:{np.var(full_kernel):.{2}}")
    plt.subplot(3, 1, 2)
    plt.title(f"First row. Mean:{np.mean(first_row):.{2}}, Var:{np.var(first_row):.{2}}")
    # plt.hist(first_row)
    # plt.subplot(4, 1, 3)
    # plt.title(f"Second row. Mean:{np.mean(second_row):.{2}}, Var:{np.var(second_row):.{2}}")
    plt.hist(second_row)
    plt.subplot(3, 1, 3)
    plt.hist(G)
    plt.title(f"G. Mean:{np.mean(G):.{2}}, Var:{np.var(G):.{2}}")
    plt.show()


if __name__ == '__main__':
    main()
