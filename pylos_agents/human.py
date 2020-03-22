""" This implements a human player as an agent to unify the framework. """
from pylos_agents.base import Agent

class Human(Agent):
    def move_list(self, game_state):
        return []