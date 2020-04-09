import numpy as np

class Encoder():
    """ The encoder follows the convention to encode all information relative to the current player, not the colors black and
    white. This facilitates the training process for neural networks."""

    @classmethod
    def get_layers(cls, game_state):
        """ This function returns a numpy array corresponding to all feature planes.
        The return dimensions are (4*<number of feature planes>, 4, 4)"""
        # my stones
        my_stones = cls._players_stones(game_state=game_state, player=game_state.current_player)
        # opponents stones
        opp_stones = cls._players_stones(game_state=game_state, player=game_state.current_player.next_player)
        # fields where I can complete a square
        my_comp_square = cls._square_completion(game_state=game_state, player=game_state.current_player)
        # fields where the opponent can complete a square
        opp_comp_square = cls._square_completion(game_state=game_state, player=game_state.current_player.next_player)
        # my free stones (not supporting anything. For raise and recover)
        my_free_stones = cls._stone_is_free(game_state=game_state, player=game_state.current_player)
        # opponents free stones (not supporting anything. For raise and recover)
        opp_free_stones = cls._stone_is_free(game_state=game_state, player=game_state.current_player.next_player)

        #Maybe also include the following to feature planes:
        # free fields in 2x2 squares where I already have 2 stones and there are two free fields left
        # free fields in 2x2 squares where the opponent already has 2 stones and there are two free fields left

        planes = [cls.ZXY_to_XYZ(p) for p in [my_stones, opp_stones, my_comp_square, opp_comp_square, my_free_stones, opp_free_stones]]
        return planes

    @classmethod
    def _players_stones(cls, game_state, player):
        """ This function returns is a feature plane that is 1 on every field on which the player has a stone
        and 0 otherwise. """
        # get players stones per layer
        is_mine = [(x == player.value) * np.ones(x.shape) for x in game_state.board]
        # turn it into a cube and return
        return cls._convert_to_cubic(is_mine)

    @classmethod
    def _square_completion(cls, game_state, player):
        """ This function returns an array that is 1 on all fields that complete a square for the player. """
        # get free fields in every layer and turn it into a cubic array
        is_free = cls._convert_to_cubic([(x == 0) * np.ones(x.shape) for x in game_state.board])
        # get coordinates of the fields and turn them into 3-tuples
        coords = np.transpose(np.where(is_free == 1))

        # for loop over coordinates: if coordinate game_state.is_support, set its field in is_free to zero
        for p in coords:
            if game_state.completes_square(tuple(p), player) is False:
                is_free[tuple(p)] = 0

        return is_free

    @classmethod
    def _stone_is_free(cls, game_state, player):
        """ This function returns an array that is 1 on all of the players that are free. This is important both for
        raising stones as well as recovering stones. """
        # get players stones per layer and turn it into a cubic array
        is_mine = cls._convert_to_cubic([(x == player.value) * np.ones(x.shape) for x in game_state.board])
        # get coordinates of the fields and turn them into 3-tuples
        coords = np.transpose(np.where(is_mine == 1))

        # for loop over coordinates: if coordinate game_state.is_support, set its field in is_mine to zero
        for p in coords:
            if game_state.is_support(tuple(p)):
                is_mine[tuple(p)] = 0

        return is_mine

    @classmethod
    def _convert_to_cubic(cls, M):
        """ This function takes a list of layers of dimensions (4,4), (3,3), (2,2), (1,1) corresponding to the
        Pylos-pyramids 4 layers and returns them in cubic form (4,4,4). It fills up empty positions w/ -1. """
        # fill up a layer w/ -1 to match the dimensions 4x4
        fill_up = lambda x: np.concatenate(
            (np.concatenate((x, np.full((4 - len(x), len(x)), 0))), np.full((4, 4 - len(x)), 0)), axis=1)

        return np.stack((M[0], fill_up(M[1]), fill_up(M[2]), fill_up(M[3])))

    @classmethod
    def ZXY_to_XYZ(cls, M):
        """interchange axes from z, x, y to x, y, z to fit the channels_last convention"""
        stack = np.swapaxes(M, 0, 1)
        return np.swapaxes(stack, 1, 2)