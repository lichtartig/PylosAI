from pylos_agents import BatchGenerator
from pylos_agents.base import PlayGames
from pylos_agents import ActorCritic, PolicyGradient, SemiRandom, Naive
from pylos_encoder import Encoder
import benchmark
import logging, os

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
    if verbose == 1: print("Starting training...")

    counter = 0
    epochs_wo_improvement = 0
    while epochs_wo_improvement < 5:
        if verbose == 1:
            print("Generating more experience data")
        states, wins, moves, advantages = play_games.play_games()
        counter += 1

        gen = BatchGenerator(encoder=encoder, states=states, wins=wins, moves=moves,
                             value_fct=advantages, epoch_size=epoch_size, output_includes_value_fct=value_fct)
        agent1.train(generator=gen, verbose=verbose)
        win1, win2 = benchmark.Benchmark(agent1, agent2)
        if verbose == 1:
            print("The agent won ", win1, " games of a total of ", win1 + win2, " against the " + str(agent2) + " AI.")
        # This corresponds to a p-value of 3%
        if win1 >= 60:
            agent1.save_weights()
            # reload the weights to improve the strength
            agent2.load_weights()
            epochs_wo_improvement = 0
        else:
            epochs_wo_improvement += 1

    # Reload agent to get the best saved weights.
    agent1.load_weights()
    win1, win2 = benchmark.Benchmark(agent1, SemiRandom(), n=1000)
    # 530 wins are a p-value of 3%
    print("Final benchmark: The agent won ", win1, " games of a total of ", win1 + win2,
          " against the SemiRandom agent.")

if __name__ == '__main__':
    # low verbosity
    logging.disable(logging.WARNING)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    # agent1 = ActorCritic()
    # agent2 = Naive()

    agent1 = PolicyGradient()
    agent2 = PolicyGradient()
    train_agent(agent1=agent1, agent2=agent2, verbose=0)
    print("No change.")

    agent1 = PolicyGradient(conv_layers=2)
    agent2 = PolicyGradient(conv_layers=2)
    train_agent(agent1=agent1, agent2=agent2, verbose=0)
    print("2 conv layers.")

    agent1 = PolicyGradient(conv_layers=4)
    agent2 = PolicyGradient(conv_layers=4)
    train_agent(agent1=agent1, agent2=agent2, verbose=0)
    print("4 conv layers.")

    agent1 = PolicyGradient(no_dense_layers=1)
    agent2 = PolicyGradient(no_dense_layers=1)
    train_agent(agent1=agent1, agent2=agent2, verbose=0)
    print("1 dense layer")

    agent1 = PolicyGradient(no_dense_layers=2)
    agent2 = PolicyGradient(no_dense_layers=2)
    train_agent(agent1=agent1, agent2=agent2, verbose=0)
    print("2 dense layers")

    agent1 = PolicyGradient(no_dense_layers=4)
    agent2 = PolicyGradient(no_dense_layers=4)
    train_agent(agent1=agent1, agent2=agent2, verbose=0)
    print("4 dense layers")