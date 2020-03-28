""" This script will run a game between to agents. All combinations of human- or bot-agents are allowed. """
import random
import time
from pylos_board.board import GameState, Player
from pylos_agents import Human, SemiRandom
from pylos_board.utilities import print_board

# this list will contain a list of game states at the end of the game
the_game = []
# this is the current game state
state = GameState.new_game()

# decide which agents play
Player1 = SemiRandom() # Human()
Player2 = SemiRandom() # Human()
# assign colors randomly
game_agents = [Player1, Player2]
random.shuffle(game_agents)
colors = dict(zip([1,-1], game_agents))

# play the game
while state.has_won() == False:
    print_board(state)
    player = colors[state.current_player.value]
    next_move = player.next_move(state)
    if next_move.is_resign:
        break
    the_game.append(state)
    state = state.apply_move(next_move)
    time.sleep(2)

print("\n\nThe game is over!")
if state.current_player == Player.white and state.has_won():
    print("White won the game!")
elif state.has_won():
    print("Black won the game!")
elif state.current_player == Player.white:
    print("White resigned.\nBlack won the game!")
else:
    print("Black resigned.\nWhite won the game!")