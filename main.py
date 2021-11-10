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

# %% Global Variables
XLIM = 30
YLIM = 30
NSTEP = 100
RHO = 0.8


# %% Gridworld Class
class GridWorld():
    def __init__(self):
        self.xlim = XLIM-1
        self.ylim = YLIM-1

    def step(self, pos, action):
        """
        (0,0) is in lower left corner
        """
        x, y = pos

        if action == "left":
            next_state = max(0, x-1), y
        elif action == "right":
            next_state = min(self.xlim, x+1), y
        elif action == "down":
            next_state = x, max(0, y-1)
        elif action == "up":
            next_state = x, min(self.ylim, y+1)
        elif action == 'left-up':
            next_state = max(0, x-1), min(y+1, self.ylim)
        elif action == 'left-down':
            next_state = max(0, x-1), max(0, y-1)
        elif action == 'right-up':
            next_state = min(self.xlim, x+1), min(y+1, self.ylim)
        elif action == 'right-down':
            next_state = min(self.xlim, x+1), max(0, y-1)
        else:
            raise ValueError

        return next_state

# %% Agent Class


class Agent():
    def __init__(self, r0, r1):
        self.action_space = [0, 180]  # Beam direction in degrees
        self.reward_0 = r0
        self.reward_1 = r1

    def get_action(self, state):
        if (self.reward_0[state[0], state[1]] >=
                self.reward_1[state[0], state[1]]):
            return 0
        else:
            return 180

# %% Functions


@numba.jit
def getCov(Lx, Ly, rho):
    sigma_squared = 1

    pos = np.zeros((Lx*Ly, 2))
    cov = np.zeros((Lx*Ly, Lx*Ly))

    for idx in range(Lx):
        for idy in range(Ly):
            pos[idx*Ly + idy, :] = [idx, idy]

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

    # Action space
    action_space = ["left", "right", "up", "down",
                    "left-up", "left-down", "right-up", "right-down"]

    # See if cov matrix is avaible:
    if os.path.exists(f"cov/cov_{XLIM}x{YLIM}_{rho}.npy"):
        print("Load covariance matrix")
        cov = np.load(f"cov/cov_{XLIM}x{YLIM}_{rho}.npy")
    else:
        print("Create covariance matrix")
        cov = getCov(XLIM, YLIM, rho)
        np.save(f"cov/cov_{XLIM}x{YLIM}_{rho}.npy", cov)

    # # See if reward matrix is avaible:
    # if os.path.exists(f"reward/reward_{XLIM}x{YLIM}_{rho}.npy"):
    #     print("Load reward matrix")
    #     reward = np.load(f"reward/reward_{XLIM}x{YLIM}_{rho}.npy")
    # else:
    #     print("Create reward matrix")
    #     reward = mnormal(mean=np.zeros(XLIM*YLIM), cov=cov, size=1)
    #     reward = reward.reshape([XLIM, YLIM])
    #     np.save(f"reward/reward_{XLIM}x{YLIM}_{rho}.npy", reward)

    reward0 = mnormal(mean=np.zeros(XLIM*YLIM), cov=cov, size=1)
    reward0 = reward0.reshape([XLIM, YLIM])

    reward1 = mnormal(mean=np.zeros(XLIM*YLIM), cov=cov, size=1)
    reward1 = reward1.reshape([XLIM, YLIM])

    # %%
    # Create agent
    agent = Agent(reward0, reward1)

    # Get enviroment
    env = GridWorld()

    # Choose N random actions based on the action space
    actions = np.random.choice(action_space, N, replace=True)

    # Create postion matrix to log path
    pos = np.zeros([len(actions)+1, 3], dtype=int)

    # Initialise starting point
    pos[0, 0:2] = np.random.randint(0, [XLIM-1, YLIM-1], dtype=int)

    # Got N steps
    for idx, action in enumerate(actions):
        pos[idx, 2] = agent.get_action(pos[idx, 0:2])
        pos[idx + 1, 0:2] = env.step(pos[idx, 0:2], action)

    # Get the last action
    pos[-1, 2] = agent.get_action(pos[-1, 0:2])

    # %% plot
    fig, ax = plt.subplots(2, 2)
    s = ax[0, 0].scatter(pos[:, 0], pos[:, 1], marker=".",
                         c=pos[:, 2], cmap="jet",
                         vmin=0, vmax=180,)

    ax[0, 0].set_xlim([-1, XLIM])
    ax[0, 0].set_ylim([-1, YLIM])
    fig.colorbar(s, ax=ax[0, 0])

    vmin = min(np.min(reward0), np.min(reward1))
    vmax = max(np.max(reward0), np.max(reward1))

    ax[1, 0].set_title("Reward 0 deg")
    s = ax[1, 0].imshow(reward0, vmin=vmin, vmax=vmax, origin="lower",
                        interpolation='None')
    ax[1, 0].set_xlim([-1, XLIM])
    ax[1, 0].set_ylim([-1, YLIM])
    fig.colorbar(s, ax=ax[1, 0])

    ax[1, 1].set_title("Reward 180 deg")
    ax[1, 1].imshow(reward1, vmin=vmin, vmax=vmax, origin="lower",
                    interpolation='None')
    ax[1, 1].set_xlim([-1, XLIM])
    ax[1, 1].set_ylim([-1, YLIM])

    fig.delaxes(ax[0, 1])
    fig.tight_layout()
