import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

from src.layers.structured_layers import get_toeplitz_like_matrix, get_circulant_matrix, ToeplitzLike
from src.elm.elms import ELM

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def main():
    main_test()


def main_test():
    test_elm_with_fastfood_layer()
    test_layer(False)

    print("Toeplitz like test was succesful!")


def test_elm_with_fastfood_layer():
    inputs = [[1, 2, 3], [3, 4, 5]]
    elm = ELM()
    elm.add_dense_layer(10, input_dim=3)
    elm.add_layer(ToeplitzLike(35, use_bias=True))
    elm.add_dense_layer(20)
    elm.add_layer(ToeplitzLike(3, use_bias=False))
    elm.compile()
    features = elm.get_transformed_features(inputs)
    assert features.shape[0] == 2
    assert features.shape[1] == 3


def test_layer(do_print=True):
    weights = [1, 2, 3]
    weights = tf.cast(weights, tf.float32)
    if do_print:
        print(f"Circulant matrice for vector [1, 2, 3]:\n {get_circulant_matrix(weights)}")
        print(f"Skew circulant matrice for vector [1, 2, 3]:\n {get_circulant_matrix(weights, is_skew=True)}")

    G_1 = [0, 5, 6, 7]
    G_2 = [1, 0, 0, 0]

    H_1 = [0, 0, 0, 1]
    H_2 = [1, 2, 3, 4]

    G = tf.cast([G_1, G_2], dtype=tf.float32)
    H = tf.cast([H_1, H_2], dtype=tf.float32)

    T = get_toeplitz_like_matrix(G, tf.reverse(H, axis=[1]))

    Z_1 = tf.cast([[0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=tf.float32)
    Z_minus_1 = tf.cast([[0, 0, 0, -1], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=tf.float32)

    displacement = tf.matmul(Z_1, T) - tf.matmul(T, Z_minus_1)
    expected_displacement = tf.matmul(G, H, transpose_a=True)

    if do_print:
        print(f"Toeplitz matrice T:\n {T}")
        print(f"Displacement operator on T:\n {displacement}")
        print(f"Expected displacement operator of T:\n{expected_displacement}")
    difference = displacement - expected_displacement
    tf.assert_greater(tf.convert_to_tensor(0.01, dtype=tf.float32), difference)

    if do_print:
        G_1 = np.random.standard_normal(5)
        G_2 = np.random.standard_normal(5)

        H_1 = np.random.standard_normal(5)
        H_2 = np.random.standard_normal(5)

        G = tf.cast([G_1, G_2], dtype=tf.float32)
        H = tf.cast([H_1, H_2], dtype=tf.float32)

        print(f"Example of toeplitz like matrice:\n{get_toeplitz_like_matrix(G, H)}")


def create_weights_plots():
    G_1 = np.random.standard_normal(10)
    G_2 = np.random.standard_normal(10)
    # G_3 = np.random.standard_normal(500)
    # G_4 = np.random.standard_normal(500)

    H_1 = np.random.standard_normal(10)
    H_2 = np.random.standard_normal(10)

    # H_3 = np.random.standard_normal(500)
    # H_4 = np.random.standard_normal(500)

    G = tf.cast([G_1, G_2], dtype=tf.float32)
    H = tf.cast([H_1, H_2], dtype=tf.float32)

    T = get_toeplitz_like_matrix(G, H)

    T = np.asarray(T).flatten()

    parameters = np.asarray([G, H]).flatten()

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.hist(T)

    plt.title(f"Weights: Mean:{np.mean(T):.{2}}, Var:{np.var(T):.{2}}")
    plt.subplot(2, 1, 2)
    plt.hist(parameters)
    plt.title(f"Parameters: Mean:{np.mean(parameters):.{2}}, Var:{np.var(parameters):.{2}}")
    plt.show()

    plt.close("all")

if __name__ == '__main__':
    main()
