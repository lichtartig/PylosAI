""" This agent will simply select a random move. It will serve as a baseline to compare future agents to. """
import random
from pylos_agents.base import Agent
from pylos_board.utilities import bottom_to_top
from pylos_board.board import Move

class SemiRandom(Agent):
    def __str__(self):
        return "SemiRandom"

    def move_list(self, game_state):
        ret = []
        # regular moves
        if game_state.stones_to_recover == 0:
            # TODO here I could test some of the feature planes to raise stones if possible

            top_to_bottom = bottom_to_top[:]
            top_to_bottom.reverse()
            # try every position starting from the top
            for pos in top_to_bottom:
                ret.append(Move.place_stone(pos))

            random.shuffle(ret)

        # recover stones
        else:
            # try every stone starting from the bottom
            for pos in bottom_to_top:
                ret.append(Move.recover_stone(pos))

            random.shuffle(ret)
            # also append pass so that agent does not accidentally resign
            ret.append(Move.pass_move())

        return ret