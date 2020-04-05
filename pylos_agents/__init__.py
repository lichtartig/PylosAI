""" This function contains different bots to play pylos. As a baseline it also contains a random playing bot. """
from pylos_agents.human import Human
from pylos_agents.random_agent import SemiRandom
from pylos_agents.naive import Naive
from pylos_agents.actor_critic import ActorCritic
from pylos_agents.policy_gradient import PolicyGradient
from pylos_agents.base import BatchGenerator