from pylos_agents.PolicyGradient import PGBatchGenerator
from pylos_agents.base import PlayGames
from pylos_agents import PolicyGradient, QLearning, ActorCritic, SemiRandom
from pylos_encoder import Encoder
import benchmark
import logging, os
import time

def train_agent(agent1, agent2, verbose=0):
    epoch_size = 60000
    encoder = Encoder()
    play_games = PlayGames(agent1=agent1, agent2=agent2, no_of_moves=epoch_size)

    # train the two agents against each other. Every time the trained agent improves, we update the opponent as well.
    # after a few epochs of no improvement, we terminate and perform a final more exact benchmark
    if verbose == 0: print("Starting training...")

    epochs_wo_improvement = 0
    while epochs_wo_improvement < 5:
        states, wins, moves = play_games.play_games()
        gen = PGBatchGenerator(agent1=agent1, agent2=agent2, encoder=encoder, states=states, wins=wins, moves=moves,
                               epoch_size=epoch_size)
        agent1.train(gen)
        win1, win2 = benchmark.Benchmark(agent1, agent2)
        if verbose == 0:
            print("The agent won ", win1, " games of a total of ", win1 + win2, " against his previous version.")
        if win1 >= 65:
            agent1.save_weights()
            # reload the weights to improve the strength
            agent2.load_weights()
            play_games = PlayGames(agent1=agent1, agent2=agent2, no_of_moves=40000)
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

    conv_layers = [1,2,4,8]
    no_of_filters = [8,16,32]
    no_dense_layers = [0, 1, 2]
    dense_dim = [64, 128]
    batch_norm = [False, True]
    dropout_rate = [0.0, 0.2]

    pooling_layers = 2 # max(conv_layers-2, 0)
    base_file = "policy_gradient_weights" # + number + ".hdf5"
    counter = 0

    print("Starting scan...")
    for c in conv_layers:
        for nof in no_of_filters:
            for ndl in no_dense_layers:
                for dd in dense_dim:
                    for bn in batch_norm:
                        for dr in dropout_rate:
                            start = time.time()
                            weight_file = base_file + str(counter) + ".hdf"
                            agent1 = PolicyGradient(conv_layers=c, no_of_filters=nof, no_dense_layers=ndl, dense_dim=dd,
                                                    batch_norm=bn, dropout_rate=dr, pooling_layers=max(c - 2, 0),
                                                    weight_file=weight_file)
                            agent2 = PolicyGradient(conv_layers=c, no_of_filters=nof, no_dense_layers=ndl, dense_dim=dd,
                                                    batch_norm=bn, dropout_rate=dr, pooling_layers=max(c - 2, 0),
                                                    weight_file=weight_file)
                            train_agent(agent1=agent1, agent2=agent2, verbose=1)
                            print(c, nof, ndl, dd, bn, dr, weight_file, str(round((time.time()-start)/60)) + " min." )
                            counter += 1

# c nof ndl dd bn dr
#1 8 0 64 True 0.0 policy_gradient_weights2.hdf Final benchmark: The agent won  644  games of a total of  1000  against the SemiRandom agent.