""" This module contains the board encoder that translates a game state to suitable features plans that can be fed
to a neural network. """

# The following feature plans seem useful:
# my stones
# opponents stones
# fields where I can complete a square
# fields where the opponent can complete a square
# free fields in 2x2 squares where I already have 2 stones and there are two free fields left
# free fields in 2x2 squares where the opponent already has 2 stones and there are two free fields left
# stones that I can lift to the next level
# stones that the opponent can lift to the next level
# stones that I can take out of the game after completing a square

# TODO Implement the encoder relative to the current player no matter it's color.