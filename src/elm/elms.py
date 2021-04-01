from tensorflow.keras.layers import Dense
from abc import ABCMeta, abstractmethod

from .base_elm import ELM
from src.layers.normalized_dense_layer import NormalizedDense
from src.layers.structured_layers import Toeplitz, Vandermonde, Circulant, FastFood, ToeplitzLike, LowRank


class ELMWithSpecificLayerClass(ELM, metaclass=ABCMeta):  # That means class is abstract.
    """ Abstract class for ELMs that have only one type of layers (with different layers params). """

    @abstractmethod
    def get_layer_class(self):
        """ Returns: class of the layer the ELM uses (Like Dense, Toeplitz). """
        pass

    def __init__(self, input_shape, number_of_layers=1, is_classifier=False, **layer_params):
        """
        Arguments:
            input_shape: Shape of the model's input.
            number_of_layers: Number of layers.
            is_classifier: Whether the model is used as classifier.
            **layers_params: kwargs for the layers parameters.
                Keys are the names of parameters.
                If value is iterable (but not string), for the layer i the value[i] will be used!
                If value is not iterable or string, this value will be used for every layer.
        """
        super().__init__(is_classifier)
        self.add_layers(number_of_layers, self.get_layer_class(), input_shape, **layer_params)
        self.compile()


def get_number_of_layers(hidden_neurons_numbers):
    return 1 if isinstance(hidden_neurons_numbers, int) else len(hidden_neurons_numbers)


class ClassicELM(ELMWithSpecificLayerClass):
    """Classical version of ELM without structures. Only Dense layers."""

    def get_layer_class(self):
        return NormalizedDense if self.normalize_weights_after_init else Dense

    def __init__(self, hidden_neurons_numbers, input_shape,
                 normalize_weights_after_init=True,
                 weight_deviation=1,
                 is_classifier=False,
                 activation="relu",
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='glorot_uniform'):
        """
        Arguments are the same as for ELMWithSpecificLayerClass and Dense layer.

        If different layers should have different parameters, give the parameters as list with the size=number of
        layers. Each element of the list param[i] has to be the parameter of the layer i.
        See **layer_params description in the superclass.

        Arguments:
            normalize_weights_after_init: if True, NormalizedDense will be used instead of Dense.
                The weights will normalized to have the deviation = weight_deviation.
            weight_deviation: see normalize_weights_after_init.
        """
        self.normalize_weights_after_init = normalize_weights_after_init
        super().__init__(input_shape, get_number_of_layers(hidden_neurons_numbers), is_classifier,
                         units=hidden_neurons_numbers,
                         normalize_weights_after_init=normalize_weights_after_init,
                         weight_deviation=weight_deviation,
                         activation=activation,
                         use_bias=use_bias,
                         kernel_initializer=kernel_initializer,
                         bias_initializer=bias_initializer)


class LowRankELM(ELMWithSpecificLayerClass):
    """ Elm with low rank and normal layers."""
    def get_layer_class(self):
        return LowRank

    def __init__(self, hidden_neurons_numbers, ranks, input_shape,
                 normalize_weights_after_init=True,
                 weight_deviation=1,
                 is_classifier=False,
                 activation="relu",
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='glorot_uniform'):
        """
        Arguments are the same as for ELMWithSpecificLayerClass and LowRank layer.

        If different layers should have different parameters, give the parameters as list with the size=number of
        layers. Each element of the list param[i] has to be the parameter of the layer i.
        See **layer_params description in the superclass.

        Arguments:
            hidden_neurons: int or array. Number of neurons in each layer.
            ranks: int or array. Rank of each layer.
                1. If you want a full rank layer, set rank equal to hidden_neurons!
                2. The program does not check, if given rank makes sense! So set rank lower or equal
                than neurons number of previous and this layer.
                3. Length of hidden_neurons must be equal length of weight_matrix_ranks.

            Example: hidden_neurons=[200, 200], weight_matrix_ranks=[200, 20] creates two layer elm, with first
                full rank layer and second low rank layer.
        """
        super().__init__(input_shape, get_number_of_layers(hidden_neurons_numbers), is_classifier,
                         units=hidden_neurons_numbers,
                         ranks=ranks,
                         normalize_weights_after_init=normalize_weights_after_init,
                         weight_deviation=weight_deviation,
                         activation=activation,
                         use_bias=use_bias,
                         kernel_initializer=kernel_initializer,
                         bias_initializer=bias_initializer)


class ToeplitzELM(ELMWithSpecificLayerClass):
    """ ELM with Toeplitz layers. """
    def get_layer_class(self):
        return Toeplitz

    def __init__(self, hidden_neurons_numbers, input_shape,
                 normalize_weights_after_init=True,
                 weight_deviation=1,
                 is_classifier=False,
                 activation="relu",
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='glorot_uniform'):
        """
        Arguments are the same as for ELMWithSpecificLayerClass and Toeplitz layer.

        If different layers should have different parameters, give the parameters as list with the size=number of
        layers. Each element of the list param[i] has to be the parameter of the layer i.
        See **layer_params description in the superclass.
        """
        super().__init__(input_shape, get_number_of_layers(hidden_neurons_numbers), is_classifier,
                         units=hidden_neurons_numbers,
                         normalize_weights_after_init=normalize_weights_after_init,
                         weight_deviation=weight_deviation,
                         activation=activation,
                         use_bias=use_bias,
                         kernel_initializer=kernel_initializer,
                         bias_initializer=bias_initializer)


class CirculantELM(ELMWithSpecificLayerClass):
    """ ELMs with Circulant layers. """
    def get_layer_class(self):
        return Circulant

    def __init__(self, hidden_neurons_numbers, input_shape,
                 normalize_weights_after_init=True,
                 weight_deviation=1,
                 is_classifier=False,
                 activation="relu",
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='glorot_uniform'):
        """
        Arguments are the same as for ELMWithSpecificLayerClass and Circulant layer.

        If different layers should have different parameters, give the parameters as list with the size=number of
        layers. Each element of the list param[i] has to be the parameter of the layer i.
        See **layer_params description in the superclass.
        """
        super().__init__(input_shape, get_number_of_layers(hidden_neurons_numbers), is_classifier,
                         units=hidden_neurons_numbers,
                         normalize_weights_after_init=normalize_weights_after_init,
                         weight_deviation=weight_deviation,
                         activation=activation,
                         use_bias=use_bias,
                         kernel_initializer=kernel_initializer,
                         bias_initializer=bias_initializer)


class VandermondeELM(ELMWithSpecificLayerClass):
    def get_layer_class(self):
        return Vandermonde

    def __init__(self, hidden_neurons_numbers, input_shape,
                 normalize_weights_after_init=True,
                 weight_deviation=1,
                 is_classifier=False,
                 drop_ones=False,
                 transpose=False,
                 activation="relu",
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='glorot_uniform'):
        """
        Arguments are the same as for ELMWithSpecificLayerClass and Vandermonde layer.

        If different layers should have different parameters, give the parameters as list with the size=number of
        layers. Each element of the list param[i] has to be the parameter of the layer i.
        See **layer_params description in the superclass.
        """
        super().__init__(input_shape, get_number_of_layers(hidden_neurons_numbers), is_classifier,
                         units=hidden_neurons_numbers,
                         normalize_weights_after_init=normalize_weights_after_init,
                         weight_deviation=weight_deviation,
                         drop_ones=drop_ones,
                         transpose=transpose,
                         activation=activation,
                         use_bias=use_bias,
                         kernel_initializer=kernel_initializer,
                         bias_initializer=bias_initializer)


class FastFoodELM(ELMWithSpecificLayerClass):
    """
    ELM that uses fast food transformation as input matrix.  More details in Layer description.
    It is recommended to chose number of neurons that is power of two.
    """
    def get_layer_class(self):
        return FastFood

    def __init__(self, hidden_neurons_numbers, input_shape,
                 normalize_weights_after_init=True,
                 weight_deviation=1,
                 is_classifier=False,
                 activation="relu",
                 use_bias=True,
                 kernel_initializer='glorot_normal',
                 bias_initializer='glorot_uniform'):
        """
        Arguments not listed below are the same as for ELMWithSpecificLayerClass and FastFood layer.

        If different layers should have different parameters, give the parameters as list with the size=number of
        layers. Each element of the list param[i] has to be the parameter of the layer i.
        See **layer_params description in the superclass.
        """
        super().__init__(input_shape, get_number_of_layers(hidden_neurons_numbers), is_classifier,
                         units=hidden_neurons_numbers,
                         normalize_weights_after_init=normalize_weights_after_init,
                         weight_deviation=weight_deviation,
                         activation=activation,
                         use_bias=use_bias,
                         kernel_initializer=kernel_initializer,
                         bias_initializer=bias_initializer)


class ToeplitzLikeELM(ELMWithSpecificLayerClass):
    """ELM that uses Toeplitz Like layers. More details in Layer description."""
    def get_layer_class(self):
        return ToeplitzLike

    def __init__(self, hidden_neurons_numbers, input_shape,
                 ranks=2,
                 normalize_weights_after_init=True,
                 weight_deviation=1,
                 is_classifier=False,
                 activation="relu",
                 use_bias=True,
                 kernel_initializer='glorot_normal',
                 bias_initializer='glorot_uniform'):
        """
        Arguments not listed below are the same as for ELMWithSpecificLayerClass and FastFood layer.

        If different layers should have different parameters, give the parameters as list with the size=number of
        layers. Each element of the list param[i] has to be the parameter of the layer i.
        See **layer_params description in the superclass.

        Arguments:
            ranks: int or array. Low displacement rank of each layer. For details see ToeplitzLike layer arguments
                Example: ranks=[2, 3], hidden_neurons=[200, 20] creates two layer elm, with first/
        """
        super().__init__(input_shape, get_number_of_layers(hidden_neurons_numbers), is_classifier,
                         units=hidden_neurons_numbers,
                         rank=ranks,
                         normalize_weights_after_init=normalize_weights_after_init,
                         weight_deviation=weight_deviation,
                         activation=activation,
                         use_bias=use_bias,
                         kernel_initializer=kernel_initializer,
                         bias_initializer=bias_initializer)