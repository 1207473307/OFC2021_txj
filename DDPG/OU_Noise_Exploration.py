import numpy as np
import random
import copy
import torch

class OU_Noise(object):
    """Ornstein-Uhlenbeck process."""
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        dx = self.theta * (self.mu - self.state) + self.sigma * np.array([np.random.normal() for _ in range(len(self.state))])
        self.state += dx
        return self.state

class Base_Exploration_Strategy(object):
    """Base abstract class for agent exploration strategies. Every exploration strategy must inherit from this class
    and implement the methods perturb_action_for_exploration_purposes and add_exploration_rewards"""
    def __init__(self, config):
        self.config = config

    def perturb_action_for_exploration_purposes(self, action_info):
        """Perturbs the action of the agent to encourage exploration"""
        raise ValueError("Must be implemented")

    def add_exploration_rewards(self, reward_info):
        """Actions intrinsic rewards to encourage exploration"""
        raise ValueError("Must be implemented")

    def reset(self):
        """Resets the noise process"""
        raise ValueError("Must be implemented")


class OU_Noise_Exploration(Base_Exploration_Strategy):
    """Ornstein-Uhlenbeck noise process exploration strategy"""
    def __init__(self, config):
        super().__init__(config)
        self.noise = OU_Noise(self.config.action_size, self.config.seed, self.config.hyperparameters["mu"],
                              self.config.hyperparameters["theta"], self.config.hyperparameters["sigma"])
        self.device = config.device

    def perturb_action_for_exploration_purposes(self, action_info):
        """Perturbs the action of the agent to encourage exploration"""
        action = action_info["action"]#.float()
        n = torch.tensor(self.noise.sample()).to(self.device)
        action += n
        # action += torch.tensor(self.noise.sample()).to(self.device)
        return action

    def add_exploration_rewards(self, reward_info):
        """Actions intrinsic rewards to encourage exploration"""
        raise ValueError("Must be implemented")

    def reset(self):
        """Resets the noise process"""
        self.noise.reset()
