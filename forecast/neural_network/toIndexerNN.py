import os
import warnings

from sklearn.exceptions import DataConversionWarning
import joblib
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPRegressor

from ..toBaseModel import IndexerModel#, load_model_interface, save_model_interface


class IndexerMLP(IndexerModel):
    """
    Initialising variables

    Args:
        length (int):  length of the input array - part of the
                       definition of the first layer shape
        numAttr (int): number of attributes - second part of
                       the definition of the first layer shape
        neurons (int): array of ints defining the number of neurons
                       in each layer and the number of layers
                       (by the length of the array) -
                       Do not include the final layer
        dropouts (list(float)): array of doubles (0 - 1) of length neurons - 1
                                defining the dropout at each level
                                - do not include the final layer
        activations (list(str)): array of strings of length neurons to
                                 define the activation of each layer
                                 - do not include the final layer
    """

    def __init__(
        self, hidden_layer_sizes=(2), activation="relu", solver="lbfgs", max_iter=200
    ):
        try:
            self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        except Exception:
            self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.max_iter = max_iter

    def build_model(self):
        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver=self.solver,
            max_iter=self.max_iter,
        )

    def compile_model(self):
        pass

    def fit_model(self, X, y, n_splits, epochs, verbose=False):
        """
        used to build and fit model on some data. uses timeseries
        cross validation in fitting the model

        Args:
            X (list(float/int)): inputs to train on
            y (list(float/int)): outputs to train on
            n_splits (int): size of batches on which to train the model
            epochs (int): number of epochs to train the model
            verbose (bool): boolean (0, 1) value to control the verbosity
                            of the fitting
        """
        tscv = TimeSeriesSplit(n_splits)
        score_t = 0
        s = 0
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            if verbose:
                print("\nNew split\n")
            warnings.filterwarnings("ignore")
            warnings.filterwarnings(action="ignore", category=DataConversionWarning)
            m = self.model.fit(X_train, y_train)
            warnings.filterwarnings("ignore")
            warnings.filterwarnings(action="ignore", category=DataConversionWarning)
            s = mean_squared_error(y_test, m.predict(X_test).reshape(-1, 1))
            if verbose:
                print("Split: ", "mean squared error: ", s)
            score_t = score_t + s
            if verbose:
                print("\nmean squared error: ", s, "\n")
        self.score_t = score_t / n_splits

    def predict(self, x):
        """return predictions by applying model to the data

        Args:
            x (list(int/float)): values on which to predict

        Returns:
            list of ints/float: predicted values
        """
        y = self.model.predict(x).reshape(-1, 1)
        return y

    def get_model(self):
        """Method to return the model

        Returns:
            Keras Model: Designed Model
        """
        return self.model

    #@load_model_interface(mode=os.environ["mode"])
    #def load_model(self, weights_path):
    #    """Method to Load model and weights to the paths specified
    #
    #    Args:
    #        weights_path (str): path to <model>.json file
    #        weights_path (str): path to <model_weights>.h5 file
    #    """
    #    self.model = joblib.load(weights_path)
    #
    #@save_model_interface(mode=os.environ["mode"])
    #def save_model(self, weights_path):
    #    """Method to save model and weights to the paths specified
    #
    #    Args:
    #        weights_path (str): path to <model>.json file
    #        weights_path (str): path to <model_weights>.h5 file
    #
    #    """
    #    joblib.dump(self.model, weights_path)

    def get_params(self):
        """Method to return the parameters of the model

        Returns:
            SK Learn Model Summary: Model Summary
        """
        return self.model.get_config()
