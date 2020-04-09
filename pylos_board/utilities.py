""" This module contains utility functions. """
import numpy as np

# all coordinates in the 4x4x4 cube that lie in the pyramid
bottom_to_top = [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3), (0, 1, 0), (0, 1, 1), (0, 1, 2), (0, 1, 3), (0, 2, 0),
                 (0, 2, 1), (0, 2, 2), (0, 2, 3), (0, 3, 0), (0, 3, 1), (0, 3, 2), (0, 3, 3), (1, 0, 0), (1, 0, 1),
                 (1, 0, 2), (1, 1, 0), (1, 1, 1), (1, 1, 2), (1, 2, 0), (1, 2, 1), (1, 2, 2), (2, 0, 0), (2, 0, 1),
                 (2, 1, 0), (2, 1, 1), (3, 0, 0)]

# all coordinates in the 4x4x4 cube that lie outside of the pyramid
off_grid = [(1, 0, 3), (1, 1, 3), (1, 2, 3), (1, 3, 3), (1, 3, 2), (1, 3, 1), (1, 3, 0), (2, 0, 3), (2, 1, 3),
            (2, 2, 3), (2, 3, 3), (2, 3, 2), (2, 3, 1), (2, 3, 0), (2, 0, 2), (2, 1, 2), (2, 2, 2), (2, 2, 1),
            (2, 2, 0), (3, 0, 3), (3, 1, 3), (3, 2, 3), (3, 3, 3), (3, 3, 2), (3, 3, 1), (3, 3, 0), (3, 0, 2),
            (3, 1, 2), (3, 2, 2), (3, 2, 1), (3, 2, 0), (3, 0, 1), (3, 1, 1), (3, 1, 0)]

def print_board(game_state):
    """ This function prints out the current board position. """
    b = game_state.board
    c = {1: '⬤', 0: '+', -1: '◯'}

    # this lambda turns one line of a numpy array into a string to print
    line_to_string = lambda y: "\t".join(map(lambda x: c[x], y))
    # build the output string
    output = ""
    for i in range(4):
        for j in range(4 - i):
            output += line_to_string(b[j][i]) + ' \t\t'
        output += '\n'

    print(output)

def print_layer(a, integer=False):
    """ This function prints 4 layers of 4x4 arrays side by side. It's useful to monitor what the encoder is doing. """
    if integer:
        a = np.int_(a)
        tot = 2
    else:
        a = np.round(a, 2)
        tot = 5

    # get equal length strings
    helper1 = lambda x: str(x) + " "*(tot-1-len(str(abs(x))))
    helper2 = lambda x: " "*(tot-len(helper1(x))) + helper1(x)

    # loop over rows
    for i in range(4):
        # loop over layers
        layer_buffer = ""
        for j in range(4):
            layer_buffer += " ".join([helper2(x) for x in a[i,:,j]]) + "\t\t\t"
        print(layer_buffer)