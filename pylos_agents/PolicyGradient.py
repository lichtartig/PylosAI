import numpy as np
from keras.models import Model
from keras.optimizers import Adadelta
from keras.layers import Input, Conv2D, BatchNormalization, Flatten, Dropout, Dense
from pylos_agents.base import Agent
from pylos_board.board import Move
from pylos_board.utilities import bottom_to_top, off_grid

# TODO will be necessary for training functions
# clipped_probs = np.clip(original_probs, min_p, max_p)
# clipped_probs = clipped_probs / np.sum(clipped_probs)

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
            x = Dropout(rate=self.dropout_rate)(x)
        # this output encodes the stones that we take out either to raise a stone or to recover after a square
        # completion. If the point lies outside of the
        recover = Dense(4**3, activation='softmax', name='recover')(x)
        # this output encodes the stones we place
        place = Dense(4**3, activation='softmax', name='place')(x)

        optimizer = Adadelta()

        self.model = Model(inputs=main_input, outputs=[recover, place])
        self.model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        self.model.summary()

        # TODO load previous weights

    def move_list(self, game_state):
        # TODO introduce a randomness parameter, i.e. set all probabilities to zero that are below a certain fraction of the best move
        """ This function uses the encoder to turn a game state into a numpy array suitable to for a neural network
         and then gets a prediction from the neural network implemented in the constructor.
         Based on this prediction it returns an ordered list of moves that is obtained by random sampling based on the
         results of the neural net. """
        # Use the encoder on the game_state and pass it to the neural net
        inp = self.encoder.get_layers(game_state)
        recover, place = self.model.predict(inp)
        # resize the results to fit to the board dimensions (4,4,4)
        recover.resize((4,4,4))
        place.resize((4,4,4))

        # get lists of probabilities where we sum up all off-grid coords into one
        coords = bottom_to_top + [(3,3,3)]
        recov_prob = [recover[(p)] for p in bottom_to_top] + [0]
        place_prob = [place[(p)] for p in bottom_to_top] + [0]
        for p in off_grid:
            recov_prob[-1] += recover[(p)]
            place_prob[-1] += place[(p)]

        # 1) the player just completed a square and there are stones to recover
        if game_state.stones_to_recover > 0:
            # get random sample according to results from neural net
            coord_indices = np.random.choice(range(len(coords)), size=len(coords), replace=False, p=recov_prob)
            moves = []
            for i in coord_indices:
                position = coords[i]
                if position in bottom_to_top:
                    moves.append(Move.recover_stone(position))
                else:
                    moves.append(Move.pass_move())
            return moves

        # 2) No stones to recover. Possible moves are raise_stone and place_stone
        else:
            # all combinations of tuples (position in recov, position in place) are distinct moves. Sample from them!
            coord_comb = [(p, q) for p in coords for q in coords]
            comb_prob = [recov_prob[recov_prob.index(p)] * place_prob[place_prob.index(q)] for (p,q) in coord_comb]
            coord_indices = np.random.choice(range(len(coord_comb)), size=len(coord_comb), replace=False, p=comb_prob)

            # make an order list of moves
            moves = []
            for (p,q) in coord_indices:
                # treat combinations that have (legal position, legal position) as raise_stone
                if game_state.is_on_grid(p) and game_state.is_on_grid(q):
                    moves.append(Move.raise_stone(p, q))

                # treat combinations that have (illegal position, legal position) as place_stone
                elif game_state.is_on_grid(q):
                    moves.append(Move.place_stone(q))

                # discard combinations that have (position, illegal position)

            return moves
