""" This module contains utility functions. """

bottom_to_top = [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3), (0, 1, 0), (0, 1, 1), (0, 1, 2), (0, 1, 3), (0, 2, 0),
                 (0, 2, 1), (0, 2, 2), (0, 2, 3), (0, 3, 0), (0, 3, 1), (0, 3, 2), (0, 3, 3), (1, 0, 0), (1, 0, 1),
                 (1, 0, 2), (1, 1, 0), (1, 1, 1), (1, 1, 2), (1, 2, 0), (1, 2, 1), (1, 2, 2), (2, 0, 0), (2, 0, 1),
                 (2, 1, 0), (2, 1, 1), (3, 0, 0)]

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