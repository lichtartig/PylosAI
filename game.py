""" This script will start a game human vs bot. """

from pylos_board.board import GameState, Move, Player

test = GameState.new_game()

a = Player.white
b = a.next_player
print(a)
print(a.value)
print(b)
print(b.value)