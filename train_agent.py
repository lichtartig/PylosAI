from pylos_agents import BatchGenerator
from pylos_agents.base import PlayGames
from pylos_agents import ActorCritic, PolicyGradient, SemiRandom
from pylos_encoder import Encoder
import benchmark
import logging, os
import time
import pickle

def load_experience(counter, play_games):
    """ This function tries to load experience data from pickled files. If it doesn't succeed, it creates this experience
    data itself and saves it."""
    counter_str = "-"+ "0"*(2-len(str(counter))) + str(counter)
    try:
        states = pickle.load(open('pickled_states'+counter_str, 'rb'))
        wins = pickle.load(open('pickled_wins'+counter_str, 'rb'))
        moves = pickle.load(open('pickled_moves'+counter_str, 'rb'))
        advantages = pickle.load(open('pickled_advantages'+counter_str, 'rb'))
    except:
        print("Generating more experience data.")
        states, wins, moves, advantages = play_games.play_games()
        pickle.dump(states, open('pickled_states'+counter_str, 'wb'))
        pickle.dump(wins, open('pickled_wins'+counter_str, 'wb'))
        pickle.dump(moves, open('pickled_moves'+counter_str, 'wb'))
        pickle.dump(advantages, open('pickled_advantages'+counter_str, 'wb'))

    return states, wins, moves, advantages

def train_agent(agent1, agent2, verbose=0):
    if str(agent1) is "ActorCritic":
        value_fct = True
    else:
        value_fct = False

    epoch_size = 2000
    encoder = Encoder()
    play_games = PlayGames(agent1=agent1, agent2=agent2, no_of_moves=epoch_size)
    # train the two agents against each other. Every time the trained agent improves, we update the opponent as well.
    # after a few epochs of no improvement, we terminate and perform a final more exact benchmark
    if verbose == 1: print("Starting training...")

    counter = 0
    epochs_wo_improvement = 0
    saved = False
    while epochs_wo_improvement < 10:
        states, wins, moves, advantages = load_experience(counter, play_games)
        counter += 1

        gen = BatchGenerator(agent1=agent1, agent2=agent2, encoder=encoder, states=states, wins=wins, moves=moves,
                             value_fct=advantages, epoch_size=epoch_size, output_includes_value_fct=value_fct)
        agent1.train(generator=gen, verbose=verbose)
        win1, win2 = benchmark.Benchmark(agent1, agent2)
        if verbose == 1:
            print("The agent won ", win1, " games of a total of ", win1 + win2, " against his previous version.")
        if win1 >= 65:
            saved = True
            agent1.save_weights()
            # reload the weights to improve the strength
            agent2.load_weights()
            epochs_wo_improvement = 0
        else:
            epochs_wo_improvement += 1

    if saved == False:
        agent1.save_weights()

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

    start = time.time()
    agent1 = ActorCritic(conv_layers=3, no_of_filters=16, no_dense_layers=1, dense_dim=64, batch_norm=False, dropout_rate=0.0)
    #agent1 = PolicyGradient(conv_layers=3, no_of_filters=16, no_dense_layers=1, dense_dim=64, batch_norm=False, dropout_rate=0.0)
    #agent2 = ActorCritic(conv_layers=c, no_of_filters=nof, no_dense_layers=ndl, dense_dim=dd, batch_norm=bn,
    #                     dropout_rate=dr, weight_file=weight_file)
    agent2 = SemiRandom()
    train_agent(agent1=agent1, agent2=agent2, verbose=1)
    print(str(round((time.time() - start) / 60)) + " min.")