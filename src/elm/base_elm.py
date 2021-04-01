from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.linear_model import Ridge, LinearRegression, RidgeClassifier
from sklearn.metrics import accuracy_score
import collections
import six
import numpy as np

from src.layers.normalized_dense_layer import NormalizedDense


class ELM:
    """Base ELM class. Provides basic structure and functionality.
    All other ELM classes should be subclasses of this class.

    Attributes:
        is_classifier: Boolean, whether the ELM is used as classifier.
        keras_model: The Tensorflow's Keras model (neural network),
            which contains all layers except for the output layer.
        output_regression: If a model (e.g. LinearRegression) is used for the output layer, this model, else None.
        output_weights_matrix: If the output weights matrix is calculated directly, this matrix, else None.
    """

    def __init__(self, is_classifier=False):
        """
        Constructs ELM object.

        Args:
            is_classifier: Boolean, whether the ELM is used as classifier.
        """
        self.is_classifier = is_classifier
        self.keras_model = Sequential()
        self.output_regression = None
        self.output_weights_matrix = None

    def add_layer(self, layer):
        """ Just adds a new layer."""
        self.keras_model.add(layer)

    def add_layers(self, number_of_layers, layer_class, first_layer_input_shape, **layers_params):
        """
        Adds many layers of the same class.

        Arguments:
            number_of_layers: Number of layers to add.
            layer_class: Class of layer (subclass of keras.layers.Layer).
            first_layer_input_shape: Shape of the first layer's input.
            **layers_params: kwargs for the layers parameters.
                Keys are the names of parameters.
                If value is iterable (but not string), for the layer i the value[i] will be used!
                If value is not iterable or string, this value will be used for every layer.
        """

        for layer_index in range(number_of_layers):
            current_layer_params = {} if layer_index > 0 else {"input_shape": first_layer_input_shape}
            for key, param in layers_params.items():
                if isinstance(param, collections.Iterable) and not isinstance(param, six.string_types):
                    current_layer_params[key] = param[layer_index]
                else:
                    current_layer_params[key] = param
            self.add_layer(layer_class(**current_layer_params))

    def add_dense_layer(self, neurons_number,
                        normalize_weights_after_init=True,
                        weight_deviation=1.,
                        activation="relu",
                        use_bias=True,
                        kernel_initializer='glorot_uniform',
                        bias_initializer='glorot_uniform',
                        **kwargs):
        """ Adds dense layer."""
        if normalize_weights_after_init:
            self.add_layer(NormalizedDense(neurons_number,
                                           weight_deviation=weight_deviation,
                                           activation=activation,
                                           use_bias=use_bias,
                                           kernel_initializer=kernel_initializer,
                                           bias_initializer=bias_initializer,
                                           **kwargs))
        else:
            self.add_layer(Dense(neurons_number,
                                 activation=activation,
                                 use_bias=use_bias,
                                 kernel_initializer=kernel_initializer,
                                 bias_initializer=bias_initializer,
                                 **kwargs))

    def compile(self):
        """
        Compiles the keras model that contains all layers except for the output layer.

        This method should be called after all layers are added and before fitting.
        By some subclasses compile occurs while initialization.
        """
        self.keras_model.compile(optimizer='adam', loss='mean_squared_error')  # The arguments are not relevant for ELM.

    def fit(self, train_x, train_y,
            regression_tool="ridge_sklearn",
            alpha=1e-3,
            rcond="warn",
            fit_intercept=False,
            normalize=False,
            copy_X=False,
            max_iter=None,
            tol=1e-3,
            class_weight=None,
            solver="auto",
            random_state=None,
            ):
        """
        Trains ELM.

        If you are using base elm class, you should add layers and compile model before using this method.

        Arguments which are not listed below are the same as for sklearn.linear_model.Ridge.fit(...).

        Arguments:
            train_x: Input that is used for training.
            train_y: Output that is used for training.
            regression_tool: How the output regression will be executed.
                "ridge_sklearn" uses sklearn.linear_model.Ridge
                "linear_sklearn" uses sklearn.linear_model.LinearRegression
                "linear_numpy" uses numpy.linalg.lstsq (least square) to calculate the output weights matrix.
            alpha: Penalty for weights. Is used only for ridge regression.
            rcond: Reciprocal condition number, Is used only for linear_numpy.
        """
        transformed_features = self.keras_model.predict(train_x)  # Also known as "hidden layer output matrix".

        if regression_tool == "ridge_sklearn":
            self.output_weights_matrix = None
            if self.is_classifier:
                self.output_regression = RidgeClassifier(alpha=alpha,
                                                         fit_intercept=fit_intercept,
                                                         normalize=normalize,
                                                         copy_X=copy_X,
                                                         max_iter=max_iter,
                                                         tol=tol,
                                                         class_weight=class_weight,
                                                         solver=solver,
                                                         random_state=random_state)
            else:
                self.output_regression = Ridge(alpha=alpha,
                                               fit_intercept=fit_intercept,
                                               normalize=normalize,
                                               copy_X=copy_X,
                                               max_iter=max_iter,
                                               tol=tol,
                                               solver=solver,
                                               random_state=random_state)
            self.output_regression.fit(transformed_features, train_y)

        elif regression_tool == "linear_sklearn":
            self.output_weights_matrix = None
            self.output_regression = LinearRegression(fit_intercept=fit_intercept,
                                                      normalize=normalize,
                                                      copy_X=copy_X)
            self.output_regression.fit(transformed_features, train_y)

        elif regression_tool == "linear_numpy":
            self.output_regression = None
            self.output_weights_matrix = np.linalg.lstsq(transformed_features, train_y, rcond=rcond)[0]  # Rcond

    def predict(self, input_x):
        """ Predicts the output for input_x. """
        transformed_features = self.keras_model.predict(input_x)
        return self.output_regression.predict(transformed_features) if self.output_weights_matrix is None \
            else np.dot(transformed_features, self.output_weights_matrix)

    def get_transformed_features(self, input_x):
        """ Returns transformed features for input_x. """
        return self.keras_model.predict(input_x)

    def calculate_rmse(self, x, true_y,
                       y_scaler=None, divisor=None):
        """
        Calculate Root Mean Square Error (RMSE).

        Arguments:
            x: The input.
            true_y: The true output.
            y_scaler: If not None, true y and predicted y will be scaled using "inverse_transform" function.
            divisor: If not None, RMSE will be divided through this number.

        """
        predicted_y = self.predict(x)
        if y_scaler is not None:
            predicted_y = y_scaler.inverse_transform(predicted_y)
            true_y = y_scaler.inverse_transform(true_y)
        if divisor is not None:
            return np.sqrt(np.mean(np.square(predicted_y - true_y))) / divisor
        return np.sqrt(np.mean(np.square(predicted_y - true_y)))

    def calculate_classification_accuracy(self, x, true_y, threshold=0, sample_weight=None, normalize=True):
        """Accuracy classification score.
        Arguments:
            x: The input.
            true_y: The true output or correct labels. Can be boolean or number matrices.
                If values of y are not boolean, threshold will be used to convert them to boolean.
            threshold: If y prediction is more than threshold, output is considered to be positive.
                Default=0. Use 0 if Ridge Classifier was used.
            normalize: Boolean. If False, return the number of correctly classified samples.
                Otherwise, return the fraction of correctly classified samples.
            sample_weight : array-like of shape (n_samples,). Sample weights.
        """
        predicted_y = self.predict(x) > threshold
        if type(true_y) is not bool:
            true_y = true_y > threshold
        return accuracy_score(true_y, predicted_y, sample_weight=sample_weight, normalize=normalize)
