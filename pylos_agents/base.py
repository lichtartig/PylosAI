""" This defines a base class for an AI-agent from which other agents may inherit. """
import random
import numpy as np
import sys
import os
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
from keras.callbacks import ModelCheckpoint
from keras.utils import Sequence
sys.stderr = stderr
from pylos_board.board import Move, GameState

class Agent:
    model = None
    weight_file = None

    def __init__(self):
        pass

    def __str__(self):
        return "base"

    def next_move(self, game_state):
        """ This function returns the next move and checks if it is valid. If not it passes to the next move in the
        list coming from move_list. If there are no moves left, it resigns.
        The reason for this implementation is to allow AI-agents not to check if a move is valid."""
        if game_state.has_stones_left == False:
            return Move.resign()
        for move in self.move_list(game_state):
            if game_state.is_valid_move(move):
                return move
        return Move.resign()

    def move_list(self, game_state):
        """ This function should return an order list of moves, which may be of length 1
        If no valid move is provided, the AI gives up. """
        return []

    def train(self, generator):
        """ This function takes a generator suitable for keras and trains the neural net on it. """
        self.model.fit_generator(generator=generator, verbose=0, use_multiprocessing=True, workers=4)

    def save_weights(self):
        """ This saves the weights to a file."""
        self.model.save_weights(self.weight_file)

    def load_weights(self):
        """ This loads the weights to a file."""
        try:
            self.model.load_weights(self.weight_file)
        except:
            pass

class BatchGenerator(Sequence):
    ''' This class serves as a data generator for keras s.t. we don't have to load all images at once.
    a game is about 40 moves. '''
    def __init__(self, agent1, agent2, encoder, states, wins, moves, batch_size=1000, epoch_size=40000):
        self.agent1 = agent1
        self.agent2 = agent2
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.encoder = encoder
        self.states = states
        self.wins = wins
        self.moves = moves

    def __len__(self):
        ''' returns the total number of batches. '''
        return int(self.epoch_size / self.batch_size)

    def __getitem__(self, index):
        """ This function should be implemented on the level of the child-BatchGenerators"""
        raise NotImplementedError


class PlayGames:
    """ This class returns a batch of moves from games played between the two agents. """
    def __init__(self, agent1, agent2, no_of_moves, calculate_advantage=False):
        self.agent1 = agent1
        self.agent2 = agent2
        self.no_of_moves = no_of_moves
        self.calculate_advantage = calculate_advantage

    def play_games(self):
        """ This function plays games of the two agents against each other and returns three lists:
        1) a list of GameStates
        2 ) a list of the winners of the corresponding games relative to the player whose turn it is (1 or -1)
        3) a list of all moves"""
        states = []
        wins = []
        moves = []
        advantage = []

        while len(states) < self.no_of_moves:
            state_buffer = [GameState.new_game()]
            move_buffer = []
            if self.calculate_advantage:
                advantage_buffer = [self.agent1.ComputeAdvantage(state_buffer[-1])]
            # assign colors randomly
            game_agents = [self.agent1, self.agent2]
            random.shuffle(game_agents)
            colors = dict(zip([1, -1], game_agents))

            # play a game
            while state_buffer[-1].has_won() == False:
                player = colors[state_buffer[-1].current_player.value]
                next_move = player.next_move(state_buffer[-1])
                if next_move.is_resign:
                    break
                move_buffer.append(next_move)
                state_buffer.append(state_buffer[-1].apply_move(next_move))
                if self.calculate_advantage:
                    advantage_buffer.append(self.agent1.ComputeAdvantage(state_buffer[-1]))

            # copy buffer to results
            states += state_buffer
            moves += move_buffer
            if self.calculate_advantage:
                advantage += advantage_buffer
            # no matter if a game ends by winning or resigning, the last state is always of the winner
            win_buffer = int(np.ceil(len(state_buffer)/2))*[-1, 1]
            if len(win_buffer) == len(states):
                wins += win_buffer
            else:
                wins += win_buffer[1:]

            return states, wins, moves, advantage