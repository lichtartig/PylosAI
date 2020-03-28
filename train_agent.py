from pylos_agents.PolicyGradient import PGBatchGenerator
from pylos_agents import PolicyGradient, QLearning, ActorCritic, SemiRandom
from pylos_encoder import Encoder
import benchmark

encoder = Encoder()
agent1 = PolicyGradient()
agent2 = PolicyGradient()
gen = PGBatchGenerator(agent1, agent2, encoder)
agent1.train(gen)
agent1.save_weights()

win1, win2 = benchmark.Benchmark(agent1, SemiRandom())
print("The agent won ", win1, " games of a total of ", win1+win2)