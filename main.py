# -*- coding: utf-8 -*-
"""
@author: Dennis Sand, Nicolai Almskou, Peter Fisker & Victor Nissen
"""

# %% Imports
import numpy as np
import numba
import matplotlib.pyplot as plt
import os
from numpy.random import multivariate_normal as mnormal
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# %% Global Variables
XLIM = 30
YLIM = 30
NSTEP = 100
RHO = 0.8


# %% Gridworld Class
class GridWorld():
    def __init__(self):
        # The limit is subtracted with 1 to avoid indices problems
        self.xlim = XLIM-1
        self.ylim = YLIM-1

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
            next_state = max(0, x-1), y
        elif step_direction == "right":
            next_state = min(self.xlim, x+1), y
        elif step_direction == "down":
            next_state = x, max(0, y-1)
        elif step_direction == "up":
            next_state = x, min(self.ylim, y+1)
        elif step_direction == 'left-up':
            next_state = max(0, x-1), min(y+1, self.ylim)
        elif step_direction == 'left-down':
            next_state = max(0, x-1), max(0, y-1)
        elif step_direction == 'right-up':
            next_state = min(self.xlim, x+1), min(y+1, self.ylim)
        elif step_direction == 'right-down':
            next_state = min(self.xlim, x+1), max(0, y-1)
        else:
            raise ValueError

        return next_state

# %% Agent Class


class Agent():
    def __init__(self, r0, r1):
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
        self.action_space = [0, 180]  # Beam direction in degrees
        self.reward_0 = r0
        self.reward_1 = r1

    def get_action(self, state):
        """
        Calculate which action is the most optimum. Currently there is two
        beam direction and it calculate which is the most optimum based
        on the reward matrix for eact direction.

        Parameters
        ----------
        state : ARRAY
            Which position on the grid you are standing on (x,y).

        Returns
        -------
        INT
            The chosen beam direction in degrees.

        """
        if (self.reward_0[state[0], state[1]] >=
                self.reward_1[state[0], state[1]]):
            return 0
        else:
            return 180

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
    sigma_squared = 1

    pos = np.zeros((Lx*Ly, 2))
    cov = np.zeros((Lx*Ly, Lx*Ly))

    # Get all the position in the grid
    for idx in range(Lx):
        for idy in range(Ly):
            pos[idx*Ly + idy, :] = [idx, idy]

    # Calculate the covariance based on the grid position
    for idx, val in enumerate(pos):
        for idx2, val2 in enumerate(pos):
            cov[idx, idx2] = sigma_squared*np.exp(-np.linalg.norm(val-val2)
                                                  / (1/(1-rho)))

    return cov


# %% main

if __name__ == '__main__':

    # number of steps:
    N = NSTEP

    # Correlation factor of reward function
    rho = RHO

    # Step space
    step_space = ["left", "right", "up", "down",
                  "left-up", "left-down", "right-up", "right-down"]

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
    reward0 = mnormal(mean=np.zeros(XLIM*YLIM), cov=cov, size=1)
    reward0 = reward0.reshape([XLIM, YLIM])

    # Reward matrix for 180 deg.
    reward1 = mnormal(mean=np.zeros(XLIM*YLIM), cov=cov, size=1)
    reward1 = reward1.reshape([XLIM, YLIM])

    # Create agent.
    agent = Agent(reward0, reward1)

    # Get enviroment
    env = GridWorld()

    # Choose N random steps based on the step space
    steps = np.random.choice(step_space, N, replace=True)

    # Create postion matrix to log path
    pos_log = np.zeros([len(steps)+1, 3], dtype=int)

    # Initialise starting point
    pos_log[0, 0:2] = np.random.randint(0, [XLIM-1, YLIM-1], dtype=int)

    # Got N steps
    for idx, step in enumerate(steps):
        pos_log[idx, 2] = agent.get_action(pos_log[idx, 0:2])
        pos_log[idx + 1, 0:2] = env.step(pos_log[idx, 0:2], step)

    # Get the last action
    pos_log[-1, 2] = agent.get_action(pos_log[-1, 0:2])

    # %% plots

    # Plots the scatterplot for the steps taken. Color indicate which action
    # it has taken at a given step
    fig, ax = plt.subplots(2, 2)
    pos_log0 = pos_log[pos_log[:, 2] == 0]
    pos_log180 = pos_log[pos_log[:, 2] == 180]

    ax[0, 0].scatter(pos_log0[:, 1], pos_log0[:, 0], marker=".",
                     vmin=0, vmax=180)
    ax[0, 0].scatter(pos_log180[:, 1], pos_log180[:, 0], marker=".",
                     vmin=0, vmax=180)

    ax[0, 0].set_xlim([-1, XLIM])
    ax[0, 0].set_ylim([-1, YLIM])
    ax[0, 0].legend(["deg 0", "deg 180"])
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

    vmin = min(np.min(reward0), np.min(reward1))
    vmax = max(np.max(reward0), np.max(reward1))

    ax[1, 0].set_title("Reward 0 deg")
    s = ax[1, 0].imshow(reward0, vmin=vmin, vmax=vmax, origin="lower",
                        interpolation='None', aspect="auto")
    ax[1, 0].set_xlim([-1, XLIM])
    ax[1, 0].set_ylim([-1, YLIM])
    fig.colorbar(s, ax=ax[1, 0], cax=axins, orientation="horizontal")

    ax[1, 1].set_title("Reward 180 deg")
    ax[1, 1].imshow(reward1, vmin=vmin, vmax=vmax, origin="lower",
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
    pos_log180 = pos_log[pos_log[:, 2] == 180]

    ax[0].scatter(pos_log0[:, 1], pos_log0[:, 0], marker=marker,
                  vmin=0, vmax=180, color="red", alpha=alpha)
    ax[0].scatter(pos_log180[:, 1], pos_log180[:, 0], marker=marker,
                  vmin=0, vmax=180, color="orange", alpha=alpha)

    ax[0].set_xlim([-1, XLIM])
    ax[0].set_ylim([-1, YLIM])
    ax[0].legend(["deg 0", "deg 180"])

    ax[1].scatter(pos_log0[:, 1], pos_log0[:, 0], marker=marker,
                  vmin=0, vmax=180, color="red", alpha=alpha)
    ax[1].scatter(pos_log180[:, 1], pos_log180[:, 0], marker=marker,
                  vmin=0, vmax=180, color="orange", alpha=alpha)

    ax[1].set_xlim([-1, XLIM])
    ax[1].set_ylim([-1, YLIM])
    ax[1].legend(["deg 0", "deg 180"])

    # Plot the generated reward functions:
    vmin = min(np.min(reward0), np.min(reward1))
    vmax = max(np.max(reward0), np.max(reward1))

    axins = inset_axes(ax[0],
                       width="100%",  # width = 100% of parent_bbox width
                       height="5%",  # height : 5%
                       loc='lower center',
                       bbox_to_anchor=(0, -0.2, 2.15, 1),
                       bbox_transform=ax[0].transAxes,
                       borderpad=0,
                       )

    ax[0].set_title("Reward 0 deg")
    s = ax[0].imshow(reward0, vmin=vmin, vmax=vmax, origin="lower",
                     interpolation='None')
    ax[0].set_xlim([-1, XLIM])
    ax[0].set_ylim([-1, YLIM])
    ax[0].autoscale(False)
    fig.colorbar(s, cax=axins, ax=ax[1], orientation="horizontal")

    ax[1].set_title("Reward 180 deg")
    ax[1].imshow(reward1, vmin=vmin, vmax=vmax, origin="lower",
                 interpolation='None')
    ax[1].set_xlim([-1, XLIM])
    ax[1].set_ylim([-1, YLIM])
    ax[1].autoscale(False)

    fig.tight_layout()
