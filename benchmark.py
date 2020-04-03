""" This script will run 100 games of one agent against the other. If it is called as main, it will print
 the results. If it is called from another module, it will return the final stats. """
import random
from pylos_board.board import GameState
from pylos_agents import Human, SemiRandom, PolicyGradient, ActorCritic

def RunGame(agent1, agent2):
    # assign colors randomly
    game_agents = [agent1, agent2]
    random.shuffle(game_agents)
    colors = dict(zip([1, -1], game_agents))

    # this is the current game state
    state = GameState.new_game()

    # play the game
    while state.has_won() == False:
        player = colors[state.current_player.value]
        next_move = player.next_move(game_state=state)
        if next_move.is_resign:
            break
        state = state.apply_move(next_move)

    if state.has_won() and colors[state.current_player.value] == agent1:
        # agent1 won
        return 1
    elif state.has_won()  and colors[state.current_player.value] == agent2:
        # agent2 won
        return 2

    elif colors[state.current_player.value] == agent1:
        # agent1 resigned
        return 2
    else:
        # agent 2 resigned
        return 1

def Benchmark(agent1, agent2, n=100):
    win1, win2 = 0,0
    for i in range(n):
        if RunGame(agent1, agent2) == 1:
            win1 += 1
        else:
            win2 += 1
    return win1, win2

if __name__ == "__main__":
    # decide which agents play
    agent1 = PolicyGradient() # SemiRandom()  # Human()
    agent2 = SemiRandom()  # Human()
    win1, win2 = 0, 0
    for i in range(1000):
        if RunGame(agent1, agent2) == 1:
            win1 += 1
        else:
            win2 += 1
    print("Player 1 (", str(agent1), ") has won", win1, " out of ", win1 + win2, " games against Player 2 (", str(agent2), ").")