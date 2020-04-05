""" This agent is an extension of the SemiRandom agent that is supposed to provide better training data for the initial phases of training.
It seems that the neural networks don't pick up on the idea of recovering or raising stones. This agent should provide more biased data.
It does the follwing:
- check if he can complete a square
- check if he can block the opponent from doing so
- check if he can raise something
- if not, perform a random move"""
import random
import copy
from pylos_agents.base import Agent
from pylos_board.utilities import bottom_to_top
from pylos_board.board import Move

class Naive(Agent):
    def __str__(self):
        return "Naive"

    def move_list(self, game_state):
        rand_order_coords = copy.deepcopy(bottom_to_top)
        random.shuffle(rand_order_coords)
        top_to_bottom = rand_order_coords[:]
        top_to_bottom.reverse()
        ret = []
        # regular moves
        if game_state.stones_to_recover == 0:
            # check if he can complete a square
            for p in rand_order_coords:
                if game_state.completes_square(p, game_state.current_player):
                    ret.append(Move.place_stone(p))

            # check if he can block the opponent from doing so
            for p in rand_order_coords:
                if game_state.completes_square(p, game_state.current_player.next_player):
                    ret.append(Move.place_stone(p))

            # check if he can raise something
            for p in bottom_to_top:
                if game_state.is_support(p) == False:
                    for q in top_to_bottom:
                        ret.append(Move.raise_stone(p, q))

            # perform a random move
            # try every position starting from the top
            for pos in top_to_bottom:
                ret.append(Move.place_stone(pos))

        # recover stones
        else:
            # try every stone starting from the bottom
            for pos in bottom_to_top:
                ret.append(Move.recover_stone(pos))

            random.shuffle(ret)
            # also append pass so that agent does not accidentally resign
            ret.append(Move.pass_move())

        return ret

    @property
    def train(self, generator):
        raise NotImplementedError("The 'SemiRandom' agent has no train method.")