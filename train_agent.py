from pylos_agents.PolicyGradient import PGBatchGenerator
from pylos_agents import PolicyGradient, QLearning, ActorCritic, SemiRandom
from pylos_encoder import Encoder
import benchmark

import logging, os

if __name__ == '__main__':
    # low verbosity
    logging.disable(logging.WARNING)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    encoder = Encoder()
    agent1 = PolicyGradient()
    agent2 = SemiRandom() # PolicyGradient()

    # train the two agents against each other. Every time the trained agent improves, we update the opponent as well.
    # after 3 epochs of no improvement, we terminate.
    print("Starting training...")
    epochs_wo_improvement = 0
    while epochs_wo_improvement < 5:
        gen = PGBatchGenerator(agent1, agent2, encoder)
        agent1.train(gen)
        win1, win2 = benchmark.Benchmark(agent1, agent2)
        win1sr, win2sr = benchmark.Benchmark(agent1, SemiRandom())
        print("The agent won ", win1, " games of a total of ", win1 + win2, " against his previous version. (", win1sr, "/", win1sr+win2sr, "against SemiRandom)")
        if win1 >= 65:
            agent1.save_weights()
            # reload the weights to improve the strength
            #agent2 = PolicyGradient()
            epochs_wo_improvement = 0
        else:
            epochs_wo_improvement += 1

    win1, win2 = benchmark.Benchmark(agent1, SemiRandom(), n=1000)
    print("Final benchmark: The agent won ", win1, " games of a total of ", win1 + win2, " against the SemiRandom agent.")