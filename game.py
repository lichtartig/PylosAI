""" This script will run a game between to agents. All combinations of human- or bot-agents are allowed. """
import random
from pylos_board.board import GameState, Player
from pylos_agents.human import Human

# this list will contain a list of game states at the end of the game
the_game = []
# this is the current game state
state = GameState.new_game()

# decide which agents play
Player1 = Human()
Player2 = Human()

# assign colors randomly
colors = {}
if random.randrange(2) == 0:
    # white
    colors[1] = Player1
    colors[-1] = Player2
else:
    colors[1] = Player2
    colors[-1] = Player1

# play the game
while state.has_won() == False:
    player = colors[state.current_player.value]
    next_move = player.next_move(state)
    if next_move.is_resign:
        if state.current_player == Player.white:
            print("White resigns.")
        else:
            print("Black resigns.")

        # append this to hold the convention that the last game state is by the winning player
        the_game.append(GameState(board=state.board, current_player=state.current_player.next_player))
        break

    the_game.append(state)
    state = state.apply_move(next_move)

print("\n\nThe game is over!")
if the_game[-1].current_player == Player.white:
    print("White won the game!")
else:
    print("Black won the game!")