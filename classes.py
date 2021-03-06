# -*- coding: utf-8 -*-
"""
@author: Dennis Sand, Nicolai Almskou, Peter Fisker & Victor Nissen
"""

import numpy as np
from collections import defaultdict


# %% Gridworld Class
class GridWorld():
    def __init__(self, xlim, ylim, reward, sigma=1):
        """
        Initialise the enviroment.

        Parameters
        ----------
        xlim : INT
            The limit for the x-axis.
        ylim : INT
            The limit for the y-axis.
        reward : MATRIX
            The reward matrix.
        sigma : FLOAT, optional
            The standard diviation for when adding randomness to the reward matrix. The default is 1.

        Returns
        -------
        None.

        """
        # The limit is subtracted with 1 to avoid indices problems
        self.xlim = xlim - 1
        self.ylim = ylim - 1
        self.sigma = sigma
        self.reward = reward

    def _get_reward(self, state, action):
        """
        Return the reward when standing in a given state and taking a given action.

        Parameters
        ----------
        state : ARRAY
            The current postion (x,y).
        action : INT
            Beam direction / current action.

        Returns
        -------
        FLOAT
            The reward for the given state and action.

        """
        mean = self.reward[state[0], state[1], action]

        return np.random.normal(mean, self.sigma, 1)

    def step(self, state, step_direction, action):
        """
        Calculate the next postion on the based on current postion and what
        the step direction is. Also return the reward for taking a chosen action
        in the given state before moving.

        Parameters
        ----------
        state : ARRAY
            The current postion (x,y).
        step_direction : STRING
            The step direction.
        action : INT
            Beam direction.

        Raises
        ------
        ValueError
            If an unkown step direction has been chosen.

        Returns
        -------
        next_state : ARRAY
            The new postion (x, y).
        reward : float
            The given reward for current state and action.

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
    def __init__(self, action_space, alpha=["constant", 0.7], eps=0.1, gamma=0.7, c=200):
        """
        Initialise the agent with the action space and the parameters for the different reinforcement algorithms

        Parameters
        ----------
        action_space : ARRAY
            An array which contains all possible actions.
        alpha : FLOAT, optional
            Parameter for RF algorithms. The default is 0.25.
        eps : FLOAT, optional
            Parameter for eps-greedy [0, 1]. The default is 0.1.
        gamma : FLOAT, optional
            Parameter used in SARSA. The default is 0.7.
        c : FLOAT, optional
            Parameter used in UCB. The default is 200.

        Returns
        -------
        None.

        """
        self.action_space = action_space  # Number of beam directions
        self.alpha_start = alpha[1]
        self.alpha_method = alpha[0]
        self.alpha = defaultdict(self._initiate_dict(alpha[1]))
        self.eps = eps
        self.gamma = gamma
        self.c = c
        self.Q = defaultdict(self._initiate_dict(0))
        self.accuracy = np.zeros(1)

    def _initiate_dict(self, value1, value2=1):
        """
        Small function used when initiating the dicts.
        For the alpha dict, value1 is alphas starting value.
        Value2 should be set to 1 as it is used to log the number of times it has been used.

        Parameters
        ----------
        value1 : FLOAT
            First value in the array.
        value2 : FLOAT, optional
            Second value in the array. The default is 1.

        Returns
        -------
        TYPE
            An iterative type which defaultdict can use to set starting values.

        """
        return lambda: [value1, value2]

    def _update_alpha(self, state, action):
        """
        Updates the alpha values if method "1/n" has been chosen

        Parameters
        ----------
        state : ARRAY
            Current position (x,y).
        action : INT
            Current action taken.

        Returns
        -------
        None.

        """
        if self.alpha_method == "1/n":
            self.alpha[state, action] = [self.alpha_start*(1/self.alpha[state, action][1]),
                                         1+self.alpha[state, action][1]]

    def greedy(self, state):
        """
        Calculate which action is expected to be the most optimum.

        Parameters
        ----------
        state : ARRAY
            Which position on the grid you are standing on (x,y).

        Returns
        -------
        INT
            The chosen action.

        """
        state = tuple(state)  # np.array not hashable

        beam_dir = self.action_space[0]
        r_est = self.Q[state, beam_dir][0]

        for action in self.action_space:
            if self.Q[state, action][0] > r_est:
                beam_dir = action
                r_est = self.Q[state, action][0]

        return beam_dir

    def e_greedy(self, state):
        """
        Return a random action in the action space based on the epsilon value.
        Else return the same value as the greedy function

        Parameters
        ----------
        state : ARRAY
            Which position on the grid you are standing on (x,y).

        Returns
        -------
        INT
            The chosen action.

        """
        if np.random.random() > self.eps:
            return self.greedy(state)
        else:
            return np.random.choice(self.action_space)

    def UCB(self, state, t):
        """
        Uses the Upper Bound Confidence method as a policy. See eq. (2.10)
        in the book:
        Reinforcement Learning - An introduction.
        Second edition by Richard S. Sutton and Andrew G. Barto

        Parameters
        ----------
        state : ARRAY
            Current position (x,y).
        t : INT
            Current time step.

        Returns
        -------
        beam_dir : INT
            Beam direction.

        """
        state = tuple(state)  # np.array not hashable

        beam_dir = self.action_space[0]
        r_est = self.Q[state, beam_dir][0] + self.c*np.sqrt(np.log(t)/self.Q[state, beam_dir][1])

        for action in self.action_space:
            r_est_new = self.Q[state, action][0] + self.c*np.sqrt(np.log(t)/self.Q[state, action][1])
            if r_est_new > r_est:
                beam_dir = action
                r_est = r_est_new

        return beam_dir

    def update(self, state, action, reward):
        """
        Update the Q table for the given state and action based on equation (2.5)
        in the book:
        Reinforcement Learning - An introduction.
        Second edition by Richard S. Sutton and Andrew G. Barto

        Parameters
        ----------
        state : ARRAY
            Which position on the grid you are standing on (x,y).
        action : INT
            The action you are taking.
        reward : MATRIX
            The reward matrix.

        Returns
        -------
        None.

        """
        state = tuple(state)

        self.Q[state, action] = [(self.Q[state, action][0] +
                                  self.alpha[state, action][0] * (reward - self.Q[state, action][0])),
                                 self.Q[state, action][1]+1]
        self._update_alpha(state, action)

    def update_sarsa(self, R, state, action, next_state, next_action):
        """
        Update the Q table for the given state and action based on equation (6.7)
        in the book:
        Reinforcement Learning - An introduction.
        Second edition by Richard S. Sutton and Andrew G. Barto

        Parameters
        ----------
        R : MATRIX
            The reward matrix.
        state : ARRAY
            Which position on the grid you are standing on (x,y).
        action : INT
            The action you are taking.
        next_state : ARRAY
            The next postion you are going to be in.
        next_action : INT
            The next action you take.

        Returns
        -------
        None.

        """
        next_state = tuple(next_state)
        state = tuple(state)
        next_Q = self.Q[next_state, next_action][0]

        self.Q[state, action] = [self.Q[state, action][0] + self.alpha[state, action][0] *
                                 (R + self.gamma * next_Q - self.Q[state, action][0]),
                                 self.Q[state, action][1]+1]
        self._update_alpha(state, action)

    def update_Q_learning(self, R, state, action, next_state):
        """
        Update the Q table for the given state and action based on equation (6.8)
        in the book:
        Reinforcement Learning - An introduction.
        Second edition by Richard S. Sutton and Andrew G. Barto

        Parameters
        ----------
        R : MATRIX
            The reward matrix.
        state : ARRAY
            Which position on the grid you are standing on (x,y).
        action : INT
            The action you are taking.
        next_state : ARRAY
            The next postion you are going to be in.

        Returns
        -------
        None.

        """
        next_state = tuple(next_state)
        state = tuple(state)
        next_action = self.greedy(next_state)
        next_Q = self.Q[next_state, next_action][0]

        self.Q[state, action] = [self.Q[state, action][0] + self.alpha[state, action][0] *
                                 (R + self.gamma * next_Q - self.Q[state, action][0]),
                                 self.Q[state, action][1]+1]
        self._update_alpha(state, action)
