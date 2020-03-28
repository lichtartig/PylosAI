from keras.models import Model
from keras.optimizers import Adadelta
from keras.layers import Input, Conv2D, BatchNormalization, Flatten, Dropout, Dense
from pylos_agents.base import Agent

class PolicyGradient(Agent):
    """ This implements policy gradient, reinforcement learning AI """

    conv_layers = 4
    batch_normalisation = False
    dropout_rate = 0.0
    no_of_filters = 32
    kernel_size = (2,2)

    def __str__(self):
        return "PolicyGradient"

    def __init__(self, encoder):
        """ This construtor compiles the neural network model based on the specs seen above. """
        self.encoder = encoder
        main_input = Input(shape=encoder.shape(), dtype='float32', name='main_input')
        x = main_input

        for i in range(self.conv_layers):
            x = Conv2D(filters=self.no_of_filters, kernel_size=self.kernel_size, padding='same',
                            activation='relu')(x)
            if self.batch_normalisation:
                x = BatchNormalization()(x)

        x = Flatten()(x)
        if self.dropout_rate > 0:
            x = Dropout(rate=self.dropout_rate)
        # this output encodes the stones that we take out either to raise a stone or to recover after a square
        # completion. If the point lies outside of the
        recover = Dense(4**3, activation='softmax', name='main_output')(x)
        # this output encodes the stones we place
        place = Dense(4**3, activation='softmax', name='main_output')(x)

        optimizer = Adadelta()

        self.model = Model(inputs=main_input, outputs={recover, place})
        self.model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        self.model.summary()

        # TODO load previous weights

    def move_list(self, game_state):
        """ This function uses the encoder to turn a game state into a numpy array suitable to for a neural network
         and then gets a prediction from the neural network implemented in the constructor.
         Based on this prediction it returns an orderd list of moves that contains a random component according to the
         parameter passed to the function. """
        # Use the encoder on the game_state and pass it to the neural net
        inp = self.encoder.get_layers(game_state)
        recover, place = self.model.predict(inp)
        # resize the results to fit to the board dimensions (4,4,4)
        recover.resize((4,4,4))
        place.resize((4,4,4))

        # Distinguish between raise_stone and place_stone by the fact whether a move is on the grid (i.e. pyramid)
        # game_state.is_on_grid(position)

        # TODO introduce a randomness parameter
        # return np.random.choice(['rock', 'paper', 'scissors'], size=3, replace=False, p=[0.5, 0.3, 0.2])

        # clipped_probs = np.clip(original_probs, min_p, max_p)
        # clipped_probs = clipped_probs / np.sum(clipped_probs)
        raise NotImplementedError