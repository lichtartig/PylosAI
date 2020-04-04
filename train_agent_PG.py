from pylos_agents import BatchGenerator
from pylos_agents.base import PlayGames
from pylos_agents import ActorCritic, PolicyGradient, SemiRandom
from pylos_encoder import Encoder
import benchmark
import logging, os
import time
import random

hyperparams = [(1, 8, 0, 64, True, 0.0, 0), (1, 8, 0, 64, True, 0.2, 1), (1, 8, 0, 64, False, 0.0, 2), (1, 8, 0, 64, False, 0.2, 3), (1, 8, 0, 128, True, 0.0, 4), (1, 8, 0, 128, True, 0.2, 5), (1, 8, 0, 128, False, 0.0, 6), (1, 8, 0, 128, False, 0.2, 7), (1, 8, 1, 64, True, 0.0, 8), (1, 8, 1, 64, True, 0.2, 9), (1, 8, 1, 64, False, 0.0, 10), (1, 8, 1, 64, False, 0.2, 11), (1, 8, 1, 128, True, 0.0, 12), (1, 8, 1, 128, True, 0.2, 13), (1, 8, 1, 128, False, 0.0, 14), (1, 8, 1, 128, False, 0.2, 15), (1, 16, 0, 64, True, 0.0, 16), (1, 16, 0, 64, True, 0.2, 17), (1, 16, 0, 64, False, 0.0, 18), (1, 16, 0, 64, False, 0.2, 19), (1, 16, 0, 128, True, 0.0, 20), (1, 16, 0, 128, True, 0.2, 21), (1, 16, 0, 128, False, 0.0, 22), (1, 16, 0, 128, False, 0.2, 23), (1, 16, 1, 64, True, 0.0, 24), (1, 16, 1, 64, True, 0.2, 25), (1, 16, 1, 64, False, 0.0, 26), (1, 16, 1, 64, False, 0.2, 27), (1, 16, 1, 128, True, 0.0, 28), (1, 16, 1, 128, True, 0.2, 29), (1, 16, 1, 128, False, 0.0, 30), (1, 16, 1, 128, False, 0.2, 31), (1, 32, 0, 64, True, 0.0, 32), (1, 32, 0, 64, True, 0.2, 33), (1, 32, 0, 64, False, 0.0, 34), (1, 32, 0, 64, False, 0.2, 35), (1, 32, 0, 128, True, 0.0, 36), (1, 32, 0, 128, True, 0.2, 37), (1, 32, 0, 128, False, 0.0, 38), (1, 32, 0, 128, False, 0.2, 39), (1, 32, 1, 64, True, 0.0, 40), (1, 32, 1, 64, True, 0.2, 41), (1, 32, 1, 64, False, 0.0, 42), (1, 32, 1, 64, False, 0.2, 43), (1, 32, 1, 128, True, 0.0, 44), (1, 32, 1, 128, True, 0.2, 45), (1, 32, 1, 128, False, 0.0, 46), (1, 32, 1, 128, False, 0.2, 47), (2, 8, 0, 64, True, 0.0, 48), (2, 8, 0, 64, True, 0.2, 49), (2, 8, 0, 64, False, 0.0, 50), (2, 8, 0, 64, False, 0.2, 51), (2, 8, 0, 128, True, 0.0, 52), (2, 8, 0, 128, True, 0.2, 53), (2, 8, 0, 128, False, 0.0, 54), (2, 8, 0, 128, False, 0.2, 55), (2, 8, 1, 64, True, 0.0, 56), (2, 8, 1, 64, True, 0.2, 57), (2, 8, 1, 64, False, 0.0, 58), (2, 8, 1, 64, False, 0.2, 59), (2, 8, 1, 128, True, 0.0, 60), (2, 8, 1, 128, True, 0.2, 61), (2, 8, 1, 128, False, 0.0, 62), (2, 8, 1, 128, False, 0.2, 63), (2, 16, 0, 64, True, 0.0, 64), (2, 16, 0, 64, True, 0.2, 65), (2, 16, 0, 64, False, 0.0, 66), (2, 16, 0, 64, False, 0.2, 67), (2, 16, 0, 128, True, 0.0, 68), (2, 16, 0, 128, True, 0.2, 69), (2, 16, 0, 128, False, 0.0, 70), (2, 16, 0, 128, False, 0.2, 71), (2, 16, 1, 64, True, 0.0, 72), (2, 16, 1, 64, True, 0.2, 73), (2, 16, 1, 64, False, 0.0, 74), (2, 16, 1, 64, False, 0.2, 75), (2, 16, 1, 128, True, 0.0, 76), (2, 16, 1, 128, True, 0.2, 77), (2, 16, 1, 128, False, 0.0, 78), (2, 16, 1, 128, False, 0.2, 79), (2, 32, 0, 64, True, 0.0, 80), (2, 32, 0, 64, True, 0.2, 81), (2, 32, 0, 64, False, 0.0, 82), (2, 32, 0, 64, False, 0.2, 83), (2, 32, 0, 128, True, 0.0, 84), (2, 32, 0, 128, True, 0.2, 85), (2, 32, 0, 128, False, 0.0, 86), (2, 32, 0, 128, False, 0.2, 87), (2, 32, 1, 64, True, 0.0, 88), (2, 32, 1, 64, True, 0.2, 89), (2, 32, 1, 64, False, 0.0, 90), (2, 32, 1, 64, False, 0.2, 91), (2, 32, 1, 128, True, 0.0, 92), (2, 32, 1, 128, True, 0.2, 93), (2, 32, 1, 128, False, 0.0, 94), (2, 32, 1, 128, False, 0.2, 95), (4, 8, 0, 64, True, 0.0, 96), (4, 8, 0, 64, True, 0.2, 97), (4, 8, 0, 64, False, 0.0, 98), (4, 8, 0, 64, False, 0.2, 99), (4, 8, 0, 128, True, 0.0, 100), (4, 8, 0, 128, True, 0.2, 101), (4, 8, 0, 128, False, 0.0, 102), (4, 8, 0, 128, False, 0.2, 103), (4, 8, 1, 64, True, 0.0, 104), (4, 8, 1, 64, True, 0.2, 105), (4, 8, 1, 64, False, 0.0, 106), (4, 8, 1, 64, False, 0.2, 107), (4, 8, 1, 128, True, 0.0, 108), (4, 8, 1, 128, True, 0.2, 109), (4, 8, 1, 128, False, 0.0, 110), (4, 8, 1, 128, False, 0.2, 111), (4, 16, 0, 64, True, 0.0, 112), (4, 16, 0, 64, True, 0.2, 113), (4, 16, 0, 64, False, 0.0, 114), (4, 16, 0, 64, False, 0.2, 115), (4, 16, 0, 128, True, 0.0, 116), (4, 16, 0, 128, True, 0.2, 117), (4, 16, 0, 128, False, 0.0, 118), (4, 16, 0, 128, False, 0.2, 119), (4, 16, 1, 64, True, 0.0, 120), (4, 16, 1, 64, True, 0.2, 121), (4, 16, 1, 64, False, 0.0, 122), (4, 16, 1, 64, False, 0.2, 123), (4, 16, 1, 128, True, 0.0, 124), (4, 16, 1, 128, True, 0.2, 125), (4, 16, 1, 128, False, 0.0, 126), (4, 16, 1, 128, False, 0.2, 127), (4, 32, 0, 64, True, 0.0, 128), (4, 32, 0, 64, True, 0.2, 129), (4, 32, 0, 64, False, 0.0, 130), (4, 32, 0, 64, False, 0.2, 131), (4, 32, 0, 128, True, 0.0, 132), (4, 32, 0, 128, True, 0.2, 133), (4, 32, 0, 128, False, 0.0, 134), (4, 32, 0, 128, False, 0.2, 135), (4, 32, 1, 64, True, 0.0, 136), (4, 32, 1, 64, True, 0.2, 137), (4, 32, 1, 64, False, 0.0, 138), (4, 32, 1, 64, False, 0.2, 139), (4, 32, 1, 128, True, 0.0, 140), (4, 32, 1, 128, True, 0.2, 141), (4, 32, 1, 128, False, 0.0, 142), (4, 32, 1, 128, False, 0.2, 143), (8, 8, 0, 64, True, 0.0, 144), (8, 8, 0, 64, True, 0.2, 145), (8, 8, 0, 64, False, 0.0, 146), (8, 8, 0, 64, False, 0.2, 147), (8, 8, 0, 128, True, 0.0, 148), (8, 8, 0, 128, True, 0.2, 149), (8, 8, 0, 128, False, 0.0, 150), (8, 8, 0, 128, False, 0.2, 151), (8, 8, 1, 64, True, 0.0, 152), (8, 8, 1, 64, True, 0.2, 153), (8, 8, 1, 64, False, 0.0, 154), (8, 8, 1, 64, False, 0.2, 155), (8, 8, 1, 128, True, 0.0, 156), (8, 8, 1, 128, True, 0.2, 157), (8, 8, 1, 128, False, 0.0, 158), (8, 8, 1, 128, False, 0.2, 159), (8, 16, 0, 64, True, 0.0, 160), (8, 16, 0, 64, True, 0.2, 161), (8, 16, 0, 64, False, 0.0, 162), (8, 16, 0, 64, False, 0.2, 163), (8, 16, 0, 128, True, 0.0, 164), (8, 16, 0, 128, True, 0.2, 165), (8, 16, 0, 128, False, 0.0, 166), (8, 16, 0, 128, False, 0.2, 167), (8, 16, 1, 64, True, 0.0, 168), (8, 16, 1, 64, True, 0.2, 169), (8, 16, 1, 64, False, 0.0, 170), (8, 16, 1, 64, False, 0.2, 171), (8, 16, 1, 128, True, 0.0, 172), (8, 16, 1, 128, True, 0.2, 173), (8, 16, 1, 128, False, 0.0, 174), (8, 16, 1, 128, False, 0.2, 175), (8, 32, 0, 64, True, 0.0, 176), (8, 32, 0, 64, True, 0.2, 177), (8, 32, 0, 64, False, 0.0, 178), (8, 32, 0, 64, False, 0.2, 179), (8, 32, 0, 128, True, 0.0, 180), (8, 32, 0, 128, True, 0.2, 181), (8, 32, 0, 128, False, 0.0, 182), (8, 32, 0, 128, False, 0.2, 183), (8, 32, 1, 64, True, 0.0, 184), (8, 32, 1, 64, True, 0.2, 185), (8, 32, 1, 64, False, 0.0, 186), (8, 32, 1, 64, False, 0.2, 187), (8, 32, 1, 128, True, 0.0, 188), (8, 32, 1, 128, True, 0.2, 189), (8, 32, 1, 128, False, 0.0, 190), (8, 32, 1, 128, False, 0.2, 191)]

def train_agent(agent1, agent2, verbose=0):
    if str(agent1) is "ActorCritic":
        value_fct = True
    else:
        value_fct = False

    epoch_size = 20000
    encoder = Encoder()
    play_games = PlayGames(agent1=agent1, agent2=agent2, no_of_moves=epoch_size)

    # train the two agents against each other. Every time the trained agent improves, we update the opponent as well.
    # after a few epochs of no improvement, we terminate and perform a final more exact benchmark
    if verbose == 0: print("Starting training...")

    epochs_wo_improvement = 0
    while epochs_wo_improvement < 20:
        states, wins, moves, advantages = play_games.play_games()
        gen = BatchGenerator(agent1=agent1, agent2=agent2, encoder=encoder, states=states, wins=wins, moves=moves,
                               advantages=advantages, epoch_size=epoch_size, value_fct=value_fct)
        agent1.train(generator=gen, verbose=verbose)
        win1, win2 = benchmark.Benchmark(agent1, agent2)
        if verbose == 0:
            print("The agent won ", win1, " games of a total of ", win1 + win2, " against his previous version.")
        if win1 >= 65:
            agent1.save_weights()
            # reload the weights to improve the strength
            agent2.load_weights()
            play_games = PlayGames(agent1=agent1, agent2=agent2, no_of_moves=epoch_size, calculate_advantage=True)
            epochs_wo_improvement = 0
        else:
            epochs_wo_improvement += 1

    # Reload agent to get the best saved weights.
    agent1.load_weights()
    win1, win2 = benchmark.Benchmark(agent1, SemiRandom(), n=1000)
    print("Final benchmark: The agent won ", win1, " games of a total of ", win1 + win2,
          " against the SemiRandom agent.")

if __name__ == '__main__':
    # low verbosity
    logging.disable(logging.WARNING)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # TODO Write Tree search AI, let it play against ActorCritic (to remain stochastic) and use it as training data
    # TODO LSTM for convolutional networks

    random.shuffle(hyperparams)

    for hp in hyperparams:
        try:
            c, nof, ndl, dd, bn, dr, counter = hp
            weight_file = "actor_critic_weights-" + (3-len(str(counter)))*"0" + str(counter) + ".hdf5"

            start = time.time()
            agent1 = PolicyGradient(conv_layers=c, no_of_filters=nof, no_dense_layers=ndl, dense_dim=dd, batch_norm=bn,
                                 dropout_rate=dr, weight_file=weight_file)
            agent2 = PolicyGradient(conv_layers=c, no_of_filters=nof, no_dense_layers=ndl, dense_dim=dd, batch_norm=bn,
                                 dropout_rate=dr, weight_file=weight_file)
            train_agent(agent1=agent1, agent2=agent2, verbose=0)
            print(counter, str(round((time.time() - start) / 60)) + " min.")
        except:
            pass