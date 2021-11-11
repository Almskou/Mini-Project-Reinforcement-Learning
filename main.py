# -*- coding: utf-8 -*-
"""
@author: Dennis Sand, Nicolai Almskou, Peter Fisker & Victor Nissen
"""

# %% Imports
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numba
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from numpy.random import multivariate_normal as mnormal

# %% Global Variables
XLIM = 10
YLIM = 10
NSTEP = 10000
RHO = 0.8
NBEAMS = 4


# %% Gridworld Class
class GridWorld():
    def __init__(self):
        # The limit is subtracted with 1 to avoid indices problems
        self.xlim = XLIM - 1
        self.ylim = YLIM - 1

    def step(self, pos, step_direction):
        """
        Calculate the next postion on the based on current postion and what
        the step direction is.

        Parameters
        ----------
        pos : ARRAY
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
        x, y = pos

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

        return next_state


# %% Agent Class
class Agent:
    def __init__(self, reward, action_space, sigma=1, alpha=0.5, eps=0.3):
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
        self.reward = reward
        self.sigma = sigma
        self.alpha = alpha
        self.eps = eps
        self.Q = defaultdict(int)

    def _get_reward(self, state, action):
        mean = self.reward[state[0], state[1], action]

        # return (np.sqrt(self.sigma) * np.random.randn(1) + self.reward[state[0], state[1], ida])
        return np.random.normal(mean, self.sigma, 1)

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
        state = tuple(state)
        beam_dir = self.action_space[0]
        r_est = self.Q[state, beam_dir]

        for idx, action in enumerate(self.action_space):
            if idx > 0:
                r_est_new = self.Q[state, action]
                if r_est_new >= r_est:
                    beam_dir = action
                    r_est = r_est_new
        return beam_dir

    def e_greedy(self, state):
        if np.random.random() > self.eps:
            return self.greedy(state)
        else:
            random_action = np.random.choice(self.action_space)
            return random_action

    def update(self, state, action):
        state = tuple(state)
        R = self._get_reward(state, action)

        self.Q[state, action] = (self.Q[state, action] +
                                 self.alpha * (R - self.Q[state, action]))


# %% Functions

@numba.jit
def getCov(Lx, Ly, rho):
    """
    Calculates the covariance matrix based on a correlation factor and
    the distiance between two points in the grid

    Parameters
    ----------
    Lx : INT
        Size of the grids x-axis.
    Ly : INT
        Size of the grids y-axis..
    rho : INT
        Correlation factor [0,1[.

    Returns
    -------
    cov : FLOAT MATRIX
        Return the covariance based on the set correlation factor.
        Has dimension (Lx*Ly, Lx*Ly)

    """
    sigma_squared = 3

    pos = np.zeros((Lx * Ly, 2))
    cov = np.zeros((Lx * Ly, Lx * Ly))

    # Get all the position in the grid
    for idx in range(Lx):
        for idy in range(Ly):
            pos[idx * Ly + idy, :] = [idx, idy]

    # Calculate the covariance based on the grid position
    for idx, val in enumerate(pos):
        for idx2, val2 in enumerate(pos):
            cov[idx, idx2] = sigma_squared * np.exp(-np.linalg.norm(val - val2)
                                                    / (1 / (1 - rho)))

    return cov


def check_result(reward, Q, beam_space):
    """
    Checks if the highest Q-value for each state, corresponds to the optimal choice

    :param reward:
    :param Q:
    :return:
    """
    optimal_choice = np.zeros_like(reward[:, :, 0])
    Q_choice = np.zeros_like(reward[:, :, 0])
    for x_coord in range(XLIM):
        for y_coord in range(YLIM):
            optimal_choice[x_coord, y_coord] = np.argmax(reward[x_coord, y_coord, :])
            beam = beam_space[0]
            q = Q[(x_coord, y_coord), beam]

            for beam_new in beam_space:
                q_new = Q[(x_coord, y_coord), beam_new]
                if q_new >= q:
                    q = q_new
                    beam = beam_new

            Q_choice[x_coord, y_coord] = beam

    result = optimal_choice == Q_choice
    return result, optimal_choice, Q_choice


def game(agent, reward, environment, step_space, n_steps, policy):
    # Choose N random steps based on the step space
    steps = np.random.choice(step_space, n_steps, replace=True)

    # Create position matrix to log path
    pos_log = np.zeros([len(steps) + 1, 3], dtype=int)

    # Initialise starting point
    pos_log[0, 0:2] = np.random.randint(0, [XLIM - 1, YLIM - 1], dtype=int)

    if policy == 'e_greedy':
        # Got N steps
        for idx, step in enumerate(steps):
            pos_log[idx, 2] = agent.e_greedy(pos_log[idx, 0:2])
            pos_log[idx + 1, 0:2] = environment.step(pos_log[idx, 0:2], step)
            agent.update(pos_log[idx, 0:2], pos_log[idx, 2])

        # Get the last action
        pos_log[-1, 2] = agent.e_greedy(pos_log[-1, 0:2])

        # return in which positions the q-values find the optimal choice
        return check_result(reward, agent.Q, agent.action_space)
    else:
        # Got N steps
        for idx, step in enumerate(steps):
            pos_log[idx, 2] = agent.greedy(pos_log[idx, 0:2])
            pos_log[idx + 1, 0:2] = environment.step(pos_log[idx, 0:2], step)
            agent.update(pos_log[idx, 0:2], pos_log[idx, 2])

        # Get the last action
        pos_log[-1, 2] = agent.greedy(pos_log[-1, 0:2])
        # return in which positions the q-values find the optimal choice
        return check_result(reward, agent.Q, agent.action_space)


# %% main

if __name__ == '__main__':

    # Correlation factor of reward function
    rho = RHO

    # Step space
    step_space = ["left", "right", "up", "down",
                  "left-up", "left-down", "right-up", "right-down"]

    # Beam Space
    beam_space = range(NBEAMS)

    # See if cov matrix is avaible:
    if os.path.exists(f"cov/cov_{XLIM}x{YLIM}_{rho}.npy"):
        print("Load covariance matrix")
        cov = np.load(f"cov/cov_{XLIM}x{YLIM}_{rho}.npy")
    else:
        print("Create covariance matrix")
        cov = getCov(XLIM, YLIM, rho)
        if not os.path.exists("cov"):
            os.makedirs("cov")
        np.save(f"cov/cov_{XLIM}x{YLIM}_{rho}.npy", cov)

    # For this example we have two beam direction 0 deg and 180 deg.
    # Therefor we need to create a reward matrix for each beam direction.
    # Reward matrix for 0 deg.
    reward = np.zeros([XLIM, YLIM, len(beam_space)])

    for idx, _ in enumerate(beam_space):
        reward_tmp = mnormal(mean=np.zeros(XLIM * YLIM), cov=cov, size=1)
        reward[:, :, idx] = reward_tmp.reshape([XLIM, YLIM])

    print('Creating agents')
    # Create agent.

    # Get environment
    env = GridWorld()

    percent_greedy = 0
    percent_e_greedy = 0
    NEPISODES = 100
    for episode in range(NEPISODES):
        print(episode)
        agent_greedy = Agent(reward, action_space=beam_space)
        agent_e_greedy = Agent(reward, action_space=beam_space)
        greedy_game = game(agent_greedy, reward, env, step_space, NSTEP, 'greedy')
        percent_greedy += np.count_nonzero(greedy_game[0]) / greedy_game[0].size

        e_greedy_game = game(agent_e_greedy, reward, env, step_space, NSTEP, 'e_greedy')
        percent_e_greedy += np.count_nonzero(e_greedy_game[0]) / e_greedy_game[0].size

    print(f'We choose the optimal choice {(percent_greedy / NEPISODES) * 100}% of the time with greedy policy')

    print(f'We choose the optimal choice {(percent_e_greedy / NEPISODES) * 100}% of the time with e_greedy policy')

    # %% plots

    # Plots the scatterplot for the steps taken. Color indicate which action
    # it has taken at a given step
    fig, ax = plt.subplots(2, 2)
    pos_log0 = pos_log[pos_log[:, 2] == 0]
    pos_log180 = pos_log[pos_log[:, 2] == 1]

    ax[0, 0].scatter(pos_log0[:, 1], pos_log0[:, 0], marker=".",
                     vmin=0, vmax=1)
    ax[0, 0].scatter(pos_log180[:, 1], pos_log180[:, 0], marker=".",
                     vmin=0, vmax=1)

    ax[0, 0].set_xlim([-1, XLIM])
    ax[0, 0].set_ylim([-1, YLIM])
    ax[0, 0].legend(["beam 0", "beam 1"])
    # fig.colorbar(s, ax=ax[0, 0])

    # Plot the generated reward functions:
    axins = inset_axes(ax[1, 0],
                       width="100%",  # width = 100% of parent_bbox width
                       height="5%",  # height : 5%
                       loc='lower center',
                       bbox_to_anchor=(0, -0.3, 2.2, 1),
                       bbox_transform=ax[1, 0].transAxes,
                       borderpad=0,
                       )

    vmin = min(np.min(reward[:, :, 0]), np.min(reward[:, :, 1]))
    vmax = max(np.max(reward[:, :, 0]), np.max(reward[:, :, 1]))

    ax[1, 0].set_title("Reward 0 deg - mean")
    s = ax[1, 0].imshow(reward[:, :, 0], vmin=vmin, vmax=vmax, origin="lower",
                        interpolation='None', aspect="auto")
    ax[1, 0].set_xlim([-1, XLIM])
    ax[1, 0].set_ylim([-1, YLIM])
    fig.colorbar(s, ax=ax[1, 0], cax=axins, orientation="horizontal")

    ax[1, 1].set_title("Reward 180 deg - mean")
    ax[1, 1].imshow(reward[:, :, 1], vmin=vmin, vmax=vmax, origin="lower",
                    interpolation='None', aspect="auto")
    ax[1, 1].set_xlim([-1, XLIM])
    ax[1, 1].set_ylim([-1, YLIM])

    fig.delaxes(ax[0, 1])
    fig.tight_layout()

    # %% Plot 2 - Best when XLIM and YLIM is low
    alpha = 0.5
    marker = "."

    fig, ax = plt.subplots(1, 2)
    pos_log0 = pos_log[pos_log[:, 2] == 0]
    pos_log180 = pos_log[pos_log[:, 2] == 1]

    ax[0].scatter(pos_log0[:, 1], pos_log0[:, 0], marker=marker,
                  vmin=0, vmax=180, color="red", alpha=alpha)
    ax[0].scatter(pos_log180[:, 1], pos_log180[:, 0], marker=marker,
                  vmin=0, vmax=180, color="orange", alpha=alpha)

    ax[0].set_xlim([-1, XLIM])
    ax[0].set_ylim([-1, YLIM])
    ax[0].legend(["beam 0", "beam 1"])

    ax[1].scatter(pos_log0[:, 1], pos_log0[:, 0], marker=marker,
                  vmin=0, vmax=1, color="red", alpha=alpha)
    ax[1].scatter(pos_log180[:, 1], pos_log180[:, 0], marker=marker,
                  vmin=0, vmax=1, color="orange", alpha=alpha)

    ax[1].set_xlim([-1, XLIM])
    ax[1].set_ylim([-1, YLIM])
    ax[1].legend(["beam 0", "beam 1"])

    # Plot the generated reward functions:
    vmin = min(np.min(reward[:, :, 0]), np.min(reward[:, :, 1]))
    vmax = max(np.max(reward[:, :, 0]), np.max(reward[:, :, 1]))

    axins = inset_axes(ax[0],
                       width="100%",  # width = 100% of parent_bbox width
                       height="5%",  # height : 5%
                       loc='lower center',
                       bbox_to_anchor=(0, -0.2, 2.15, 1),
                       bbox_transform=ax[0].transAxes,
                       borderpad=0,
                       )

    ax[0].set_title("Reward beam 0 - mean")
    s = ax[0].imshow(reward[:, :, 0], vmin=vmin, vmax=vmax, origin="lower",
                     interpolation='None')
    ax[0].set_xlim([-1, XLIM])
    ax[0].set_ylim([-1, YLIM])
    ax[0].autoscale(False)
    fig.colorbar(s, cax=axins, ax=ax[1], orientation="horizontal")

    ax[1].set_title("Reward beam 1 - mean")
    ax[1].imshow(reward[:, :, 1], vmin=vmin, vmax=vmax, origin="lower",
                 interpolation='None')
    ax[1].set_xlim([-1, XLIM])
    ax[1].set_ylim([-1, YLIM])
    ax[1].autoscale(False)

    fig.tight_layout()
    plt.show()
