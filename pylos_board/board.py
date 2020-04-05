import enum
import copy
import numpy as np
from pylos_board.utilities import bottom_to_top

class GameState():
    """ This class stores the current board position as well whose players turn it is. """
    def __init__(self, board, current_player, stones_to_recover=0):
        self.board = board
        self.current_player = current_player
        self.stones_to_recover = stones_to_recover

    def apply_move(self, move):
        """ This function takes a Move-object as a parameter and returns a new game-state object corresponding to the
        next position. """
        new_board = copy.deepcopy(self.board)
        if move.is_pass:
            return GameState(board=new_board, current_player=self.current_player.next_player)

        elif move.is_recover:
            z, x, y = move.current_position
            new_board[z][x,y] = 0
            if self.stones_to_recover > 1:
                next_player = self.current_player
            else:
                next_player = self.current_player.next_player
            return GameState(board=new_board, current_player=next_player, stones_to_recover=self.stones_to_recover-1)

        elif move.is_raise:
            z, x, y = move.current_position
            new_board[z][x, y] = 0
            z, x, y = move.new_position
            new_board[z][x, y] = self.current_player.value
            if self.completes_square(move.new_position, self.current_player):
                return GameState(board=new_board, current_player=self.current_player, stones_to_recover=2)
            else:
                return GameState(board=new_board, current_player=self.current_player.next_player)

        else:
            z, x, y = move.new_position
            new_board[z][x, y] = self.current_player.value
            if self.completes_square(move.new_position, self.current_player):
                return GameState(board=new_board, current_player=self.current_player, stones_to_recover=2)
            else:
                return GameState(board=new_board, current_player=self.current_player.next_player)


    def is_valid_move(self, move):
        """ This function checks if a given move is allowed. It is a wrapper for the private functions below."""
        # If stones_to_recover is larger than zero, the only allowed moves are recover or pass
        if self.stones_to_recover > 0 and move.is_recover == False and move.is_pass == False:
            return False
        elif self.stones_to_recover == 0 and (move.is_recover or move.is_pass):
            return False

        # check if new position is on grid
        if move.new_position != None and self.is_on_grid(move.new_position) is False:
            return False

        # check if current position is on grid
        if move.current_position != None and self.is_on_grid(move.current_position) is False:
            return False

        # Check if the new position is free and has support
        if move.new_position != None and (self.get_value(move.new_position) != 0 or self._has_support(move.new_position) == False):
            return False

        # Check if a stone may be taken out or if it supports stones above it
        if move.current_position != None and self.is_support(move.current_position):
            return False

        # Check if the stone to be taken out is of the current players color
        if move.current_position != None and self.get_value(move.current_position) != self.current_player.value:
            return False

        # If the move raises a stone check if the layer is higher.
        if move.is_raise and move.new_position[0] <= move.current_position[0]:
            return False

        # Make sure to check if a move is legal taking into account that taking out the stone at the current position
        # might reduce the number of legal positions on the layer above.
        if move.is_raise and move.current_position in self._supporting_stones(move.new_position):
            return False

        # If the player has no stones left to place, he has to pass
        if self.has_stones_left() == 0 and move.is_pass == False:
            return False

        return True

    def has_won(self):
        """ This function returns True if the player has won. """
        return self.board[3][0,0] == self.current_player.value

    def is_support(self, position):
        """ This function checks whether a stone at a position given as a 3-tuple (layer, x-coord., y-coord.) is
        supporting stones in a layer above it. """
        list_of_pos_values = map(lambda x: self.get_value(x) != 0, self._stones_on_top(position))
        return any(list_of_pos_values)

    def _has_support(self, position):
        """ This function checks whether a position has supporting stones in the layer below it. """
        # get list of coordinates of supporting positions, then check component-wise if non-empty
        list_of_pos_values = map(lambda x: self.get_value(x) != 0, self._supporting_stones(position))
        return all(list_of_pos_values)

    def has_stones_left(self):
        """ Check if there are stones left to place for this player. """
        stones_in_layer = lambda x: (x == self.current_player.value).sum()
        return sum(map(stones_in_layer, self.board)) < 16

    def completes_square(self, position, player):
        """ Checks whether a given completes a square. """
        my_color = lambda x: self.get_value(x) == player.value
        z, x, y = position
        squares = [[(z, x+1, y), (z, x, y+1), (z, x+1, y+1)],
                   [(z, x, y+1), (z, x-1, y), (z, x-1, y+1)],
                   [(z, x, y-1), (z, x-1, y), (z, x-1, y-1)],
                   [(z, x, y-1), (z, x+1, y), (z, x+1, y-1)]]
        for s in squares:
            try:
                if all(map(my_color, s)):
                    return True
            except IndexError:
                pass

        return False

    def _supporting_stones(self, position):
        """ Given a position, this returns a list of the coordinates of the supporting stones."""
        z, x, y = position
        if z == 0:
            return []
        else:
            return [(z-1, x, y), (z-1, x+1, y), (z-1, x, y+1), (z-1, x+1, y+1)]

    def _stones_on_top(self, position):
        """ Given a position, this returns a list of the coordinates on top of it. """
        z, x, y = position
        ret = []
        if z < 3:
            if 0 < x and 0 < y:
                ret.append((z+1, x-1, y-1))
            if 0 < x and y < 3-z:
                ret.append((z+1, x-1, y))
            if x < 3-z and 0 < y:
                ret.append((z+1, x, y-1))
            if x < 3-z and y < 3-z:
                ret.append((z+1, x, y))

        return ret

    def is_on_grid(self, position):
        return True if position in bottom_to_top else False

    def get_value(self, position):
        """ Given a position this function returns the value at that position (-1: black, 0: nothing, 1: white) """
        return self.board[position[0]][position[1], position[2]]

    @classmethod
    def new_game(cls):
        """ This returns a new GameState object corresponding to an empty board. """
        board = [np.zeros((4,4)), np.zeros((3,3)), np.zeros((2,2)), np.zeros((1,1))]
        return GameState(board=board, current_player=Player.white, stones_to_recover=0)

class Move():
    """ This class represents all possible moves a player can make. """
    def __init__(self, new_position=None, current_position=None, is_raise=False, is_recover=False, is_pass=False, is_resign=False):
        assert is_raise + is_recover + is_pass <= 1
        if is_recover == False and is_pass == False and is_resign == False:
            assert type(new_position) is type((0,))
        # This is not elif, b/c of the raise move which passes both if loops
        if is_raise or is_recover:
            assert type(current_position) is type((0,))

        self.new_position = new_position
        self.current_position = current_position
        self.is_raise = is_raise
        self.is_recover = is_recover
        self.is_pass = is_pass
        self.is_resign = is_resign

    @classmethod
    def place_stone(cls, new_position):
        """ This function performs the standard move of placing a stone of the current players color at the indicated
        position given in the form of a 3-tuple that indicates (layer, x-coord., y-coord.) """
        return Move(new_position=new_position)

    @classmethod
    def raise_stone(cls, current_position, new_position):
        """ This function lets the current player raise a stone from a lower layer to a higher one.
        Both positions are given as 3-tuples (layer, x-coord., y-coord.)."""
        return Move(new_position=new_position, current_position=current_position, is_raise=True)

    @classmethod
    def recover_stone(cls, current_position):
        """ This function lets the current player recover stones after completing a colored 2x2 square of the same color.
        It checks if the player is allowed to recover a stone and also if the stone it wants to recover supports stones
        in higher layers.
        The position of the stone to be taken out is given as a 3-tuple (layer, x-coord., y-coord.)."""
        return Move(current_position=current_position, is_recover=True)

    @classmethod
    def pass_move(cls):
        """ This allows for the case that a player only wants to take out one stone after completing a square. """
        return Move(is_pass=True)

    @classmethod
    def resign(cls):
        return Move(is_resign=True)


class Player(enum.Enum):
    """ This class simply provides a small enum to track whose turn it is in the GameState class"""
    white = 1
    black = -1

    @property
    def next_player(cls):
        return Player.black if cls == Player.white else Player.white