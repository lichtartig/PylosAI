import random
import sys
import os
import numpy as np
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
from keras.models import Model
from keras.optimizers import Adadelta
from keras.layers import Input, Conv2D, Flatten, Dense, concatenate, MaxPooling2D, Dropout, BatchNormalization
sys.stderr = stderr
from pylos_agents.base import Agent, BatchGenerator
from pylos_board.board import Move
from pylos_board.utilities import bottom_to_top, off_grid
from pylos_encoder import Encoder

class PolicyGradient(Agent):
    """ This implements policy gradient, reinforcement learning AI."""

    def __init__(self, eps=1e-3, conv_layers=4, no_of_filters=8, kernel_size=(2, 2), pool_size=(2, 2), pooling_layers=2,
                 no_dense_layers=2, dense_dim=64, batch_norm=False, dropout_rate=0.0, weight_file = "policy_gradient_weights.hdf5"):
        """ This constructor compiles the neural network model based on the specs seen above.
        The parameter eps encodes the clipping applied to probabilities to keep the process stochastic.
        probabilities can be as low as (eps) and as high as (1-eps)"""
        self.encoder = Encoder()
        self.eps = eps
        self.conv_layers = conv_layers
        self.no_of_filters = no_of_filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.pooling_layers = pooling_layers
        self.no_dense_layers = no_dense_layers
        self.dense_dim = dense_dim
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate
        self.weight_file = weight_file

        # get inputs
        inputs = []
        inputs.append(Input(shape=(4,4,4), dtype='float32', name='my_stones'))
        inputs.append(Input(shape=(4,4,4), dtype='float32', name='opp_stones'))
        inputs.append(Input(shape=(4,4,4), dtype='float32', name='my_comp_square'))
        inputs.append(Input(shape=(4,4,4), dtype='float32', name='opp_comp_square'))
        inputs.append(Input(shape=(4,4,4), dtype='float32', name='my_free_stones'))
        inputs.append(Input(shape=(4,4,4), dtype='float32', name='opp_free_stones'))


        # We treat all inputs in independent streams while passing through the convolutional networks, because there is
        # no relevant spatial information between different feature planes
        layers = [inputs]
        for i in range(self.conv_layers):
            nxt_layer = []
            for l in layers[-1]:
                tmp = Conv2D(filters=self.no_of_filters, kernel_size=self.kernel_size, padding='same', activation='relu')(l)
                if i == self.conv_layers-self.pooling_layers: tmp = MaxPooling2D(pool_size=self.pool_size)(tmp)
                if self.batch_norm: tmp = BatchNormalization()(tmp)
                if i == self.conv_layers-1: tmp = Flatten()(tmp)
                nxt_layer.append(tmp)
            layers.append(nxt_layer)
        # concatenate streams to proceed
        x = concatenate(layers[-1])


        # Dense part of the network
        for i in range(self.no_dense_layers):
            x = Dense(self.dense_dim, activation='relu')(x)
            if self.dropout_rate > 0.0: x = Dropout(rate=self.dropout_rate)(x)


        # Output encodes the stones that we take out either to raise a stone or to recover after a square
        # completion. If the point lies outside of the
        recover = Dense(4**3, activation='softmax', name='recover')(x)
        # this output encodes the stones we place
        place = Dense(4**3, activation='softmax', name='place')(x)


        # compile the model
        optimizer = Adadelta()
        self.model = Model(inputs=inputs, outputs=[recover, place])
        self.model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy')

        # load weights if possible
        try:
            self.model.load_weights(self.weight_file)
        except:
            pass

    def __str__(self):
        return "PolicyGradient"

    def move_list(self, game_state):
        """ This function uses the encoder to turn a game state into a numpy array suitable to for a neural network
         and then gets a prediction from the neural network implemented in the constructor.
         Based on this prediction it returns an ordered list of moves that is obtained by random sampling based on the
         results of the neural net.
         The parameter eps encodes that a probability may be as low as (eps) and as high as (1-eps) """
        # Use the encoder on the game_state and pass it to the neural net
        inputs = [l[None,:,:,:] for l in self.encoder.get_layers(game_state)]
        recover, place = self.model.predict(inputs)
        # resize the results to fit to the board dimensions (4,4,4)
        recover.resize((4, 4, 4))
        place.resize((4, 4, 4))

        # get lists of probabilities where we sum up all off-grid coords into one
        coords = bottom_to_top + [(3, 3, 3)]
        recov_prob = np.array([recover[(p)] for p in bottom_to_top] + [0])
        place_prob = np.array([place[(p)] for p in bottom_to_top] + [0])
        for p in off_grid:
            recov_prob[-1] += recover[(p)]
            place_prob[-1] += place[(p)]
        # clip and renormalize to avoid that a move becomes impossible or the only move. This disables learning.
        recov_prob = np.clip(recov_prob, self.eps, 1-self.eps)
        place_prob = np.clip(recov_prob, self.eps, 1-self.eps)
        recov_prob = recov_prob / recov_prob.sum()
        place_prob = place_prob / place_prob.sum()

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
            comb_prob = np.array([recov_prob[coords.index(p)] * place_prob[coords.index(q)] for (p,q) in coord_comb])
            # clip and renormalize to avoid that a move becomes impossible or the only move. This disables learning.
            comb_prob = np.clip(comb_prob, self.eps, 1 - self.eps)
            comb_prob = comb_prob / comb_prob.sum()
            coord_indices = np.random.choice(range(len(coord_comb)), size=len(coord_comb), replace=False, p=comb_prob)

            # make an order list of moves
            moves = []
            for i in coord_indices:
                p, q = coord_comb[i]
                # treat combinations that have (legal position, legal position) as raise_stone
                if game_state.is_on_grid(p) and game_state.is_on_grid(q):
                    moves.append(Move.raise_stone(p, q))

                # treat combinations that have (illegal position, legal position) as place_stone
                elif game_state.is_on_grid(q):
                    moves.append(Move.place_stone(q))

                # discard combinations that have (position, illegal position)

            return moves

class PGBatchGenerator(BatchGenerator):
    """ This Generator returns the data in the form necessary for the PolicyGradient agent above. Most of its functions
    are inherited from the base class. """

    def __getitem__(self, index):
        """ Turns the data received from self._play_games into a format that fits the PG-agent."""
        inp = [[], [], [], [], [], []]
        recov_targets = []
        place_targets = []
        for i in range(len(self.moves)):
            feature_planes = self.encoder.get_layers(self.states[i])
            for i in range(len(inp)):
                inp[i].append(feature_planes[i])

            if self.moves[i].is_raise:
                new_p = self.moves[i].new_position
                cur_p = self.moves[i].current_position
            elif self.moves[i].is_pass:
                new_p = random.choice(off_grid)
                cur_p = random.choice(off_grid)
            elif self.moves[i].is_recover:
                new_p = random.choice(off_grid)
                cur_p = self.moves[i].current_position
            else:
                new_p = self.moves[i].new_position
                cur_p = random.choice(off_grid)

            rcv_tmp = np.zeros((4,4,4))
            plc_tmp = np.zeros((4,4,4))
            # This is where the actual Policy as in PolicyGradient is implemented.
            rcv_tmp[cur_p] = self.wins[i]
            plc_tmp[new_p] = self.wins[i]
            recov_targets.append(rcv_tmp.flatten())
            place_targets.append(plc_tmp.flatten())

        # turn the components of inp into numpy arrays
        inp = [np.array(i) for i in inp]

        return inp, [np.array(recov_targets), np.array(place_targets)]