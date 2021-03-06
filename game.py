""" This script will run a game between to agents. All combinations of human- or bot-agents are allowed. """
import random
import time
from pylos_board.board import GameState, Player
from pylos_agents import Human, SemiRandom, PolicyGradient, ActorCritic, Naive
from pylos_board.utilities import print_board, print_layer


# this is the current game state
state = GameState.new_game()

# decide which agents play
#agent1 = ActorCritic(conv_layers=3, no_of_filters=16, no_dense_layers=1, dense_dim=64, batch_norm=False, dropout_rate=0.0) # PolicyGradient() # SemiRandom() # Human()
agent1 = Naive()
agent2 = Human() # SemiRandom() # PolicyGradient(encoder)
# assign colors randomly
game_agents = [agent1, agent2]
random.shuffle(game_agents)
colors = dict(zip([1,-1], game_agents))

# play the game
while state.has_won() == False:
    print_board(state)
    player = colors[state.current_player.value]
    next_move = player.next_move(state)
    if next_move.is_recover:
        print("recover", state.stones_to_recover)
    if next_move.is_resign:
        break
    state = state.apply_move(next_move)
    if str(agent1) != "Human" and str(agent2) != "Human":
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