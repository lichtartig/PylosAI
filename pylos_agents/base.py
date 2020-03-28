""" This defines a base class for an AI-agent from which other agents may inherit. """
from pylos_board.board import Move

class Agent:
    def __str__(self):
        return "base"

    def __init__(self):
        pass

    def next_move(self, game_state):
        """ This function returns the next move and checks if it is valid. If not it passes to the next move in the
        list coming from move_list. If there are no moves left, it resigns.
        The reason for this implementation is to allow AI-agents not to check if a move is valid."""
        for move in self.move_list(game_state):
            if game_state.is_valid_move(move):
                return move
        return Move.resign()

    def move_list(self, game_state):
        """ This function should return an order list of moves, which may be of length 1
        If no valid move is provided, the AI gives up. """
        return []