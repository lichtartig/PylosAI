import enum

class GameState():
    """ This class stores the current board position as well whose players turn it is. """
    def __init__(self):
        raise NotImplementedError

    def apply_move(self, move):
        """ This function takes a Move-object as a parameter and returns a new game-state object corresponding to the
        next position. Before it does so it checks whether the move is legal.
        1) Check if the position has supporting stones below it and if the game is won.
        2) If the move raises a stone, it checks if the move is legal taking into account whether the stone does not
        support other stones and if the layer is higher.
        Make sure to check if a move is legal taking into account that taking out the stone at the current position
        might reduce the number of legal positions on the layer above.
        3) If the move is to recover a stone, check if the player is allowed to do so and also if the stone to be taken
        out supports stones in the layer above it. """

        raise NotImplementedError

    def stone_is_support(self, position):
        """ This function checks whether a stone at a position given as a 3-tuple (layer, x-coord., y-coord.) is
        supporting stones in a layer above it. """
        raise NotImplementedError

    def has_support(self, position):
        """ This function checks whether a position has supporting stones in the layer below it. """
        raise NotImplementedError

    def get_hash(self):
        """ To make a lookup of GameStates in a dictionary easier, this returns a hash of the current state. """
        raise NotImplementedError


class Move():
    """ This class represents all possible moves a player can make. All checks whether a move is legal are performed in
    the GameState class. """
    def __init__(self, new_position=None, current_position=None, is_raise=False, is_recover=False):
        assert is_raise * is_recover is 0
        if is_recover is False:
            assert type(new_position) is type((0,))
        # This is not elif, b/c of the raise move which passes both if loops
        if is_raise or is_recover:
            assert type(current_position) is type((0,))

        self.new_position = new_position
        self.current_position = current_position
        self.is_raise = is_raise
        self.is_recover = is_recover

    @classmethod
    def place_stone(self, new_position):
        """ This function performs the standard move of placing a stone of the current players color at the indicated
        position given in the form of a 3-tuple that indicates (layer, x-coord., y-coord.) """
        return Move(new_position=new_position)

    @classmethod
    def raise_stone(self, current_position, new_position):
        """ This function lets the current player raise a stone from a lower layer to a higher one.
        Both positions are given as 3-tuples (layer, x-coord., y-coord.)."""
        return Move(new_position=new_position, current_position=current_position, is_raise=True)

    @classmethod
    def recover_stone(self, current_position):
        """ This function lets the current player recover stones after completing a colored 2x2 square of the same color.
        It checks if the player is allowed to recover a stone and also if the stone it wants to recover supports stones
        in higher layers.
        The position of the stone to be taken out is given as a 3-tuple (layer, x-coord., y-coord.)."""
        return Move(current_position=current_position, is_recover=True)

class Player(enum.Enum):
    """ This class simply provides a small enum to track whose turn it is in the GameState class"""
    white = 0
    black = 1

    @property
    def next_player(self):
        return Player.black if self.Player == white else Player.white