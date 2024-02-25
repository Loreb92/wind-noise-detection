from tensorflow import keras


class WindClassifier(keras.Model):
    """ Shallow Neural Network classifier.
    Ref.: https://keras.io/api/models/model/#model-class
    """
    def __init__(self, input_dim, output_dim=1, activation='sigmoid'):
        """
        Parameters
        ----------
        input_dim : int
            Dimension of the input layer.
        output_dim : int
            Dimension of the output layer.
        activation : str
            Activation function.
        dropout : float
            Dropout rate.
        """
        super(WindClassifier, self).__init__()

        # define the model
        self.model = keras.Sequential()
        self.model.add(keras.layers.Dense(output_dim, input_shape=(input_dim,), activation=activation))


    def call(self, inputs, training=False):
        """ Forward pass.
        """
        return self.model(inputs, training=training)
    

def build_ffnn_model(input_dim, output_dim, activation, learning_rate):
    """ Build a feed-forward neural network, define the optimizer, loss function, and the metrics.
    """

    keras.backend.clear_session()
    model = WindClassifier(input_dim, output_dim=output_dim, activation=activation)

    # define loss and optimizer
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = keras.losses.BinaryCrossentropy()

    # define metrics
    roc_auc_metric = keras.metrics.AUC(
        num_thresholds=200,
        curve='ROC',
        summation_method='interpolation',
        name='ROC-AUC'
    )

    pr_auc_metric = keras.metrics.AUC(
        num_thresholds=200,
        curve='PR',
        summation_method='interpolation',
        name='PR-AUC'
    )

    # compile
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=[roc_auc_metric, pr_auc_metric])

    return model
        