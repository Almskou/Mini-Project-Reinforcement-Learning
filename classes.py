# -*- coding: utf-8 -*-
"""
@author: Dennis Sand, Nicolai Almskou, Peter Fisker & Victor Nissen
"""

import numpy as np
from collections import defaultdict


# %% Gridworld Class
class GridWorld():
    def __init__(self, xlim, ylim, reward, sigma=1):
        # The limit is subtracted with 1 to avoid indices problems
        self.xlim = xlim - 1
        self.ylim = ylim - 1
        self.sigma = sigma
        self.reward = reward

    def _get_reward(self, state, action):
        mean = self.reward[state[0], state[1], action]

        return np.random.normal(mean, self.sigma, 1)

    def step(self, state, step_direction, action):
        """
        Calculate the next postion on the based on current postion and what
        the step direction is.

        Parameters
        ----------
        state : ARRAY
            The current postion (x,y).
        step_direction : STRING
            The step direction.

        Raises
        ------
        ValueError
            If an unkown step direction has been chosen.

        Returns
        -------
        next_state : ARRAY
            The new postion.

        """
        x, y = state

        if step_direction == "left":
            next_state = max(0, x - 1), y
        elif step_direction == "right":
            next_state = min(self.xlim, x + 1), y
        elif step_direction == "down":
            next_state = x, max(0, y - 1)
        elif step_direction == "up":
            next_state = x, min(self.ylim, y + 1)
        elif step_direction == 'left-up':
            next_state = max(0, x - 1), min(y + 1, self.ylim)
        elif step_direction == 'left-down':
            next_state = max(0, x - 1), max(0, y - 1)
        elif step_direction == 'right-up':
            next_state = min(self.xlim, x + 1), min(y + 1, self.ylim)
        elif step_direction == 'right-down':
            next_state = min(self.xlim, x + 1), max(0, y - 1)
        else:
            raise ValueError

        reward = self._get_reward(state, action)

        return next_state, reward


# %% Agent Class
class Agent:
    def __init__(self, action_space, sigma=1, alpha=0.5, eps=0.3, gamma=1):
        """
        Initialise the agent with the reward matrices. One for each direction.

        Parameters
        ----------
        r0 : FLOAT MATRIX
            Reward matrix for the 0 degree beam direction.
        r1 : FLOAT MATRIX
            Reward matrix for the 180 degree beam direction..

        Returns
        -------
        None.

        """
        self.action_space = action_space  # Number of beam directions
        self.alpha = alpha
        self.eps = eps
        self.gamma = gamma
        self.Q = defaultdict(int)

    def greedy(self, state):
        """
        Calculate which action is the most optimum. Currently there is two
        beam direction and it calculate which is the most optimum based
        on the reward matrix for each direction.

        Parameters
        ----------
        state : ARRAY
            Which position on the grid you are standing on (x,y).

        Returns
        -------
        INT
            The chosen beam direction in degrees.

        """
        state = tuple(state)  # np.array not hashable

        beam_dir = self.action_space[0]
        r_est = self.Q[state, beam_dir]

        for action in self.action_space:
            if self.Q[state, action] > r_est:
                beam_dir = action
                r_est = self.Q[state, action]

        return beam_dir

    def e_greedy(self, state):
        if np.random.random() > self.eps:
            return self.greedy(state)
        else:
            return np.random.choice(self.action_space)

    def update(self, state, action, reward):
        state = tuple(state)

        self.Q[state, action] = (self.Q[state, action] +
                                 self.alpha * (reward - self.Q[state, action]))

    def update_sarsa(self, R, state, action, next_state, next_action):
        next_state = tuple(state)
        state = tuple(state)
        next_Q = self.Q[next_state, next_action]

        self.Q[state, action] += self.alpha * (R + self.gamma*next_Q - self.Q[state, action])
