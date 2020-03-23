""" This implements a human player as an agent to unify the framework. """
from pylos_agents.base import Agent
from pylos_board.board import Move, Player

class Human(Agent):
    def __init__(self):
        self.name = input("Type in your name: ")
        self.color_dict = {1:"white", -1:"black"}

    def move_list(self, game_state):
        self._print_board(game_state)
        print("It's your turn, ", self.name, "(", self.color_dict[game_state.current_player.value], ")!")
        if game_state.stones_to_recover == 1:
            print("You may recover a stone from the board.")
        elif game_state.stones_to_recover == 2:
            print("You may recover two stones from the board.")

        turn = input("Your move: ")
        move = self._interprete_turn(game_state, turn)

        # check if valid move and otherwise ask for another one
        while game_state.is_valid_move(move) == False:
            print("This is not a valid move. Please provide a different one!")
            turn = input("Your move: ")
            move = self._interprete_turn(game_state, turn)

        return [move]

    def _print_board(self, game_state):
        """ This function prints out the current board position. """
        b = game_state.board
        c = {1: '⬤', 0: '+', -1: '◯'}

        # this lambda turns one line of a numpy array into a string to print
        line_to_string = lambda y: "\t".join(map(lambda x: c[x], y))
        # build the output string
        output = ""
        for i in range(4):
            for j in range(4-i):
                output += line_to_string(b[j][i]) + ' \t\t'
            output += '\n'

        print(output)

    def _interprete_turn(self, game_state, turn):
        """ This function turns a string provided by the user into a Move-object. """
        letter_to_number = {"A":0, "B":1, "C":2, "D": 3}
        position = lambda x: (letter_to_number[x[0]], int(x[1]), int(x[2]))

        if turn == "resign":
            return Move.resign()

        # place stone and recover
        elif len(turn) == 3:
            try:
                if game_state.get_value(position(turn)) != 0:
                    return Move.recover_stone(position(turn))
                else:
                    return Move.place_stone(position(turn))
            # this is to ensure that an illegal entry such as A43 does not raise an error but is later recognised illegal
            except IndexError:
                return Move.place_stone(position(turn))

        # raise stone
        else:
            return Move.raise_stone(current_position=position(turn[:3]), new_position=position(turn[4:]))
