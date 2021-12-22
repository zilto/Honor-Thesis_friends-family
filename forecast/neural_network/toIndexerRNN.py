"""
Recurrent Neural Netwrorks
==========================
This Class consists of all the classes used for building
Recurrent neural network models.
Any variations of Recurrent networks should be added here.
"""
import os

import numpy as np
from keras import backend as K
from keras.layers import (
    GRU,
    LSTM,
    Activation,
    Add,
    Bidirectional,
    Concatenate,
    Dense,
    Dropout,
    Flatten,
    Input,
    Lambda,
    Multiply,
    Reshape,
    SimpleRNN,
)
from keras.models import Sequential, load_model
from keras.optimizers import SGD, Adadelta, Adagrad, Adam, Adamax, Nadam, RMSprop
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from ..toBaseModel import IndexerModel #, load_model_interface, save_model_interface

optimizers = {}
optimizers["SGD"] = SGD
optimizers["RMSprop"] = RMSprop
optimizers["Adagrad"] = Adagrad
optimizers["Adadelta"] = Adadelta
optimizers["Adam"] = Adam
optimizers["Adamax"] = Adamax
optimizers["Nadam"] = Nadam


class IndexerRNN(IndexerModel):
    """
    Base RNN class Implementation.

    Args:
        input_shape tuple(int, int): A tuple of (length of sequence,
                                     number of features)
        neurons List[int]: array of ints defining the number of neurons
                           in each layer and the number of layers
                           (by the length of the array)
                           - Do not include the final layer
        dropouts List[float]: array of doubles (0 - 1) of length
                              neurons - 1 defining the dropout at
                              each level - do not include the final layer
        activations List[str]: array of strings of length neurons to
                               define the activation of each layer
                               - do not include the final layer
    """

    def __init__(self, input_shape, neurons, dropouts, activations):
        print("initialising RNN")
        super().__init__()
        self.input_shape = input_shape
        self.neurons = neurons
        self.dropouts = dropouts
        self.activations = activations
        self.model = Sequential()
        self._validate_params()

    def _validate_params(self):
        """
        validates the params of the model graph
        """
        if len(self.neurons) != len(self.dropouts) + 1:
            raise ValueError(
                """dim of dropouts must
                    be one less than dim of neurons"""
            )
        if len(self.neurons) != len(self.activations):
            raise ValueError("""dim of activationsnot the same dim of neurons""")

    def build_model(self, *args, **kwargs):
        pass

    def compile_model(
        self, loss="mse", optimizer="rmsprop", metrics=["mean_squared_error"], **kwargs
    ):
        """
        Method to compile the model using the parameters
        given by the inputs to the method

        Args:
            loss (str): the loss functions to use in the compilation
            optimizer (str): the optimizer to use in the compilation
            metrics (list(str)): list of metrics to use in the compilation
            shuffle (bool): boolean to shuffle values in fitting
        """
        if type(optimizer) == dict:
            name = optimizer.pop("name")
            print(name)
            optimizer = optimizers[name](**optimizer)
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics, **kwargs)

    def fit_model(self, x, y, epochs, n_splits, batch_size, verbose, **kwargs):
        """
        Method to fit the model to data provided

        Args:
            xtrain (list(float/int)): inputs to train on
            ytrain (list(float/int)): outputs to train on
            epochs (int): number of epochs to train the model
            batchSize (int): size of batches on which to train the model
            verbose (bool): boolean (0, 1) value to control the verbosity
                            of the fitting
        """
        tscv = TimeSeriesSplit(n_splits)
        score_t = 0
        s = 0
        for train_index, test_index in tscv.split(x):
            X_train, X_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            if verbose:
                print("\nNew split\n")
            history = self.model.fit(
                X_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                verbose=verbose,
                **kwargs
            )
            s = mean_squared_error(y_test, self.model.predict(X_test).reshape(-1, 1))
            if verbose:
                print("Split: ", "mean squared error: ", s)
            score_t = score_t + s
            if verbose:
                print("\nmean squared error: ", s, "\n")
        self.score_t = score_t / n_splits

        return history

    def predict(self, X):
        """
        Method to provide predictions from the model

        Args:
            x (list(ints/float)): values on which to predict

        Returns:
            list of ints/float:predicted values
        """
        return self.model.predict(X)

    def get_params(self):
        """Method to return the parameters of the model

        Returns:
            Keras Model Summary: Model Summary
        """
        return self.model.get_config()

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
    #        weights_path (str): path to <model_weights>.h5 file
    #    """
    #    self.model = load_model(weights_path)
    #
    #@save_model_interface(mode=os.environ["mode"])
    #def save_model(self, weights_path):
    #    """Method to save model and weights to the paths specified
    #
    #    Args:
    #        weights_path (str): target path to <model_weights>.h5 file
    #
    #    """
    #    self.model.save(weights_path)


class IndexerSimpleRNN(IndexerRNN):
    """
    SimpleRNN model using RNN Layers
    """

    def __init__(self, input_shape, neurons, dropouts, activations):
        super().__init__(input_shape, neurons, dropouts, activations)

    def build_model(self):
        """
        Method to build the model graph
        """

        self.model.add(
            SimpleRNN(
                self.neurons[0],
                return_sequences=True,
                activation=self.activations[0],
                input_shape=self.input_shape,
            )
        )

        self.model.add(Dropout(self.dropouts[0]))

        for i in np.arange(start=1, stop=len(self.neurons) - 1):
            self.model.add(
                SimpleRNN(
                    self.neurons[i],
                    return_sequences=True,
                    activation=self.activations[i],
                )
            )
            self.model.add(Dropout(self.dropouts[i - 1]))

        # Add the final layer
        self.model.add(
            SimpleRNN(
                self.neurons[len(self.neurons) - 1],
                return_sequences=True,
                activation=self.activations[len(self.activations) - 1],
            )
        )
        self.model.add(Dense(1, activation="linear",))


class IndexerLSTM(IndexerRNN):
    """
    Long Short Term Memory class abstraction

    Args:
        r_dropouts (list(float)): list of doubles (0 - 1) of length
                                  neurons - 1 defining the dropout
                                  at each level - do not include
                                  the final layer
    """

    def __init__(self, input_shape, neurons, dropouts, activations, r_dropouts):
        super().__init__(input_shape, neurons, dropouts, activations)
        self.r_dropouts = r_dropouts
        self._validate_child_params()

    def _validate_child_params(self):
        if len(self.neurons) != len(self.r_dropouts):
            print
            raise ValueError(
                """dim of recurrent dropouts array same as\
                        the dim of neurons"""
            )

    def build_model(self):
        """
        Method to build the model graph.
        """

        self.model.add(
            LSTM(
                self.neurons[0],
                return_sequences=True,
                activation=self.activations[0],
                recurrent_dropout=self.r_dropouts[0],
                input_shape=self.input_shape,
            )
        )

        self.model.add(Dropout(self.dropouts[0]))

        for i in np.arange(start=1, stop=len(self.neurons)):
            self.model.add(
                LSTM(
                    self.neurons[i],
                    return_sequences=True,
                    activation=self.activations[i],
                    recurrent_dropout=self.r_dropouts[i],
                )
            )
            self.model.add(Dropout(self.dropouts[i - 1]))

        # Add the final layer
        self.model.add(
            LSTM(
                self.neurons[len(self.neurons) - 1],
                return_sequences=True,
                activation=self.activations[len(self.activations) - 1],
                recurrent_dropout=self.r_dropouts[len(self.r_dropouts) - 1],
            )
        )
        self.model.add(Flatten())
        self.model.add(Dense(1, activation="linear",))


class IndexerBDLSTM(IndexerRNN):
    """
    Implementation of Bidirectional LSTM
    """

    def __init__(self, input_shape, neurons, dropouts, activations):
        super().__init__(input_shape, neurons, dropouts, activations)

    def build_model(self):
        """Method to build the model graph
        """
        self.model.add(
            Bidirectional(
                LSTM(
                    self.neurons[0],
                    return_sequences=True,
                    activation=self.activations[0],
                ),
                input_shape=self.input_shape,
            )
        )
        self.model.add(Dropout(self.dropouts[0]))
        for i in np.arange(start=1, stop=len(self.neurons)):
            if i < len(self.neurons):
                self.model.add(
                    LSTM(
                        self.neurons[i],
                        return_sequences=True,
                        activation=self.activations[i],
                    )
                )
            else:
                self.model.add(
                    LSTM(
                        self.neurons[i],
                        return_sequences=False,
                        activation=self.activations[i],
                    )
                )

        self.model.add(Flatten())
        self.model.add(Dense(1, activation="linear"))


class IndexerGRU(IndexerRNN):
    """
    Gated Recurrent Network Implementation
    """

    model = None

    def __init__(self, input_shape, neurons, dropouts, activations):
        print("initiialising GRU")
        super().__init__(input_shape, neurons, dropouts, activations)

    def build_model(self):
        """
        Method to build the model graph
        """
        self.model.add(
            GRU(
                self.neurons[0],
                return_sequences=True,
                activation=self.activations[0],
                input_shape=self.input_shape,
            )
        )
        self.model.add(Dropout(self.dropouts[0]))

        for i in np.arange(start=1, stop=len(self.neurons)):
            self.model.add(
                GRU(
                    self.neurons[i],
                    return_sequences=True,
                    activation=self.activations[i],
                )
            )
            self.model.add(Dropout(self.dropouts[i - 1]))

        # Add the final layer
        self.model.add(
            GRU(
                self.neurons[len(self.neurons) - 1],
                return_sequences=True,
                activation=self.activations[len(self.activations) - 1],
            )
        )
        self.model.add(Flatten())
        self.model.add(Dense(1, activation="linear",))


##### SP models ###########


class IndexerRNNDense(IndexerRNN):
    """
    Implementation of RNN + Dense Layer Model
    input_shape: shape of input data, (n_memory_steps, n_in_features)
    output_shape: shape of output data, (n_forcast_steps, n_out_features)
    cell: cell in the RNN part, 'SimpleRNN' / 'LSTM' / 'GRU'
    neurons: number of hidden cell unit in RNN part, integer, e.g. 100
    dense_units: units of the hidden dense layers, a tuple, e.g, (20,30)
    """

    def __init__(
        self,
        input_shape,
        output_shape,
        neurons,
        dropouts,
        activations,
        cell,
        dense_units,
    ):
        super().__init__(input_shape, neurons, dropouts, activations)

        self.output_shape = output_shape
        self.dense_units = dense_units
        self.cell = cell
        self._validate_child_params()

    def _validate_child_params(self):
        assert self.cell in ["SimpleRNN", "LSTM", "GRU"]
        assert type(self.neurons) == int
        assert type(self.dense_units) == tuple

    def build_model(self):
        """Method to build the model graph
        """
        x_in = Input(self.input_shape)
        if self.cell == "SimpleRNN":
            x = SimpleRNN(units=self.neurons)(x_in)
        elif self.cell == "LSTM":
            x = LSTM(units=self.neurons)(x_in)
        elif self.cell == "GRU":
            x = GRU(units=self.neurons)(x_in)

        if self.dense_units != None:
            for i, n_units in enumerate(self.dense_units):
                x = Dense(n_units, activation=self.activations[i])(x)
                x = Dropout(self.dropouts[i])(x)
        x = Dense(np.prod(self.output_shape))(x)
        x_out = Reshape((self.output_shape))(x)
        self.model = Model(inputs=x_in, outputs=x_out)


class IndexerRNNHiddenDense(IndexerRNN):
    """
    Implementation of RNN + Dense Layer Model, here the hidden states go to the dense layers
    input_shape: shape of input data, (n_memory_steps, n_in_features)
    output_shape: shape of output data, (n_forcast_steps, n_out_features)
    cell: cell in the RNN part, 'SimpleRNN' / 'LSTM' / 'GRU'
    cell_units: number of hidden cell unit in RNN part, integer, e.g. 100
    dense_units: units of the hidden dense layers, a tuple, e.g, (20,30)
    """

    def __init__(
        self,
        input_shape,
        output_shape,
        neurons,
        dropouts,
        activations,
        cell,
        dense_units,
    ):
        super().__init__(input_shape, neurons, dropouts, activations)

        self.output_shape = output_shape
        self.dense_units = dense_units
        self.cell = cell
        self._validate_child_params()

        def _validate_child_params(self):
            assert self.cell in ["SimpleRNN", "LSTM", "GRU"]
            assert type(self.dense_units) == tuple

    def build_model(self):
        """Method to build the model graph
        """
        x_in = Input(self.input_shape)
        if self.cell == "SimpleRNN":
            x = SimpleRNN(units=self.neurons, return_sequences=True)(x_in)
        elif self.cell == "LSTM":
            x = LSTM(units=self.neurons, return_sequences=True)(x_in)
        elif self.cell == "GRU":
            x = GRU(units=self.neurons, return_sequences=True)(x_in)

        if self.dense_units != None:
            for i, n_units in enumerate(self.dense_units):
                x = Dense(n_units, activation=self.activations[i])(x)
                x = Dropout(self.dropouts[i])(x)
        x = Reshape((-1, self.output_shape[-1] * self.cell_units))(x)
        x = Dense(np.prod(self.output_shape))(x)
        x_out = Reshape((self.output_shape))(x)
        self.model = Model(inputs=x_in, outputs=x_out)


class IndexerDilatedConv(IndexerRNN):
    """
    ===== Model Architecture ========
    16 dilated causal convolutional blocks
    Preprocessing and postprocessing (time distributed) fully connected layers (convolutions with filter width 1): 16 output units
    32 filters of width 2 per block
    Exponentially increasing dilation rate with a reset (1, 2, 4, 8, ..., 128, 1, 2, ..., 128)
    Gated activations
    Residual and skip connections
    2 (time distributed) fully connected layers to map sum of skip outputs to final output

    neurons : (any int .ie 8) so the the model looks like [1,2,4,..2^8,1,2,4,..2^8]
    Note : Some activations are fixed and not meant to be changed.
    """

    def __init__(
        self, input_shape, neurons, dropouts, activations, n_filters, filter_width
    ):
        super().__init__(input_shape, neurons, dropouts, activations)

        self.n_filters = n_filters
        self.filter_width = filter_width
        self._validate_child_params()

    def _validate_child_params(self):
        assert type(self.n_filters) == int
        assert type(self.filter_width) == int

    def build_model(self):
        """Method to build the model graph
        """
        print("Model creation happens here ............................")
        dilation_rates = [2 ** i for i in range(self.neurons)] * 2
        history_seq = Input(shape=(self.input_shape))
        x = history_seq

        skips = []
        for i, dilation_rate in enumerate(dilation_rates):

            # preprocessing - equivalent to time-distributed dense
            x = Conv1D(16, 1, padding="same", activation=activation[0])(x)

            # filter convolution
            x_f = Conv1D(
                filters=self.n_filters,
                kernel_size=self.filter_width,
                padding="causal",
                dilation_rate=dilation_rate,
            )(x)

            # gating convolution
            x_g = Conv1D(
                filters=self.n_filters,
                kernel_size=self.filter_width,
                padding="causal",
                dilation_rate=dilation_rate,
            )(x)

            # multiply filter and gating branches
            z = Multiply()([Activation("tanh")(x_f), Activation("sigmoid")(x_g)])

            # postprocessing - equivalent to time-distributed dense
            z = Conv1D(16, 1, padding="same", activation=activation[0])(z)

            # residual connection
            x = Add()([x, z])

            # collect skip connections
            skips.append(z)

        # add all skip connection outputs
        out = Activation("relu")(Add()(skips))

        # final time-distributed dense layers
        out = Conv1D(128, 1, padding="same")(out)
        out = Activation("relu")(out)
        out = Dropout(dropouts[0])(out)
        out = Conv1D(1, 1, padding="same")(out)

        def slice(x, seq_length):
            return x[:, -seq_length:, :]

        pred_seq_train = Lambda(slice, arguments={"seq_length": 60})(out)

        self.model = Model(history_seq, pred_seq_train)


class IndexerEncoderDecoderRNN(IndexerRNN):
    """
    Implementation of Encoder Decoder RNN model without teacher forcing
    input_shape: shape of input data, (n_memory_steps, n_in_features)
    output_shape: shape of output data, (n_forcast_steps, n_out_features)
    cell: cell in the RNN part, 'SimpleRNN' / 'LSTM' / 'GRU'
    cell_units: number of hidden cell unit in RNN part, integer, e.g. 100
    dense_units: units of the hidden dense layers, a tuple, e.g, (20,30)
    """

    def __init__(self, input_shape, output_shape, neurons, dropouts, activations, cell):
        super().__init__(input_shape, neurons, dropouts, activations)

        self.output_shape = output_shape
        self.cell = cell
        self._validate_child_params()

        def _validate_child_params(self):
            assert self.cell in ["SimpleRNN", "LSTM", "GRU"]

    def build_model(self):
        """Method to build the model graph
        """
        if self.cell == "LSTM":
            # declare encoder and decoder objects
            encoder = LSTM(units=self.neurons, return_state=True)
            decoder = LSTM(units=self.neurons, return_sequences=True, return_state=True)
            decoder_dense = Dense(self.output_shape[-1])

            # data flow
            encoder_input = Input(self.input_shape)
            encoder_output, state_h, state_c = encoder(encoder_input)
            encoder_state = [state_h, state_c]

            decoder_input = Input((1, self.output_shape[-1]))

            # initial input and state for iteration
            iter_input = decoder_input
            iter_state = encoder_state
            all_output = []

            for _ in range(output_shape[0]):
                # Run the decoder on one timestep, output == state_h since only one time step
                output, state_h, state_c = decoder(iter_input, initial_state=iter_state)
                output = decoder_dense(output)

                # Store the current prediction (we will concatenate all predictions later)
                all_output.append(output)

                # Reinject the outputs and state for the next loop iteration
                iter_input = output
                iter_state = [state_h, state_c]

        elif self.cell == "SimpleRNN":
            # declare encoder and decoder objects
            encoder = SimpleRNN(units=self.neurons, return_state=True)
            decoder = SimpleRNN(
                units=self.neurons, return_sequences=True, return_state=True
            )
            decoder_dense = Dense(self.output_shape[-1])

            # data flow
            encoder_input = Input(self.input_shape)
            encoder_output, state_h = encoder(encoder_input)
            encoder_state = state_h

            decoder_input = Input((1, self.output_shape[-1]))

            # initial input and state for iteration
            iter_input = decoder_input
            iter_state = encoder_state
            all_output = []

            for _ in range(self.output_shape[0]):
                # Run the decoder on one timestep, output == state_h since only one time step
                output, state_h = decoder(iter_input, initial_state=iter_state)
                output = decoder_dense(output)

                # Store the current prediction (we will concatenate all predictions later)
                all_output.append(output)

                # Reinject the outputs and state for the next loop iteration
                iter_input = output
                iter_state = state_h

        elif self.cell == "GRU":
            # declare encoder and decoder objects
            encoder = GRU(units=self.neurons, return_state=True)
            decoder = GRU(units=self.neurons, return_sequences=True, return_state=True)
            decoder_dense = Dense(self.output_shape[-1])

            # data flow
            encoder_input = Input(self.input_shape)
            encoder_output, state_h = encoder(encoder_input)
            encoder_state = state_h

            decoder_input = Input((1, self.output_shape[-1]))

            # initial input and state for iteration
            iter_input = decoder_input
            iter_state = encoder_state
            all_output = []

            for _ in range(output_shape[0]):
                # Run the decoder on one timestep, output == state_h since only one time step
                output, state_h = decoder(iter_input, initial_state=iter_state)
                output = decoder_dense(output)

                # Store the current prediction (we will concatenate all predictions later)
                all_output.append(output)

                # Reinject the outputs and state for the next loop iteration
                iter_input = output
                iter_state = state_h

        # Concatenate all predictions
        decoder_output = Lambda(lambda x: K.concatenate(x, axis=1))(all_output)
        self.model = Model([encoder_input, decoder_input], decoder_output)
