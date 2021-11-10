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
XLIM = 50
YLIM = 50


# %% Gridworld Class
class GridWorld():
    def __init__(self):
        self.xlim = XLIM
        self.ylim = YLIM

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
    N = 100

    # Correlation factor of reward function
    rho = 0.99

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

    # See if reward matrix is avaible:
    if os.path.exists(f"reward/reward_{XLIM}x{YLIM}_{rho}.npy"):
        print("Load reward matrix")
        reward = np.load(f"reward/reward_{XLIM}x{YLIM}_{rho}.npy")
    else:
        print("Create reward matrix")
        reward = mnormal(mean=np.zeros(XLIM*YLIM), cov=cov, size=1)
        reward = reward.reshape([XLIM, YLIM])
        np.save(f"reward/reward_{XLIM}x{YLIM}_{rho}.npy", reward)

    # Get enviroment
    env = GridWorld()

    # Choose N random actions based on the action space
    actions = np.random.choice(action_space, N, replace=True)

    # Create postion matrix to log path
    pos = np.zeros([len(actions)+1, 2])

    # Initialise starting point
    pos[0, :] = np.random.randint(0, [XLIM, YLIM])

    # Got N steps
    for idx, action in enumerate(actions):
        pos[idx + 1, :] = env.step(pos[idx, :], action)

    # %% plot
    fig, ax = plt.subplots()
    ax.scatter(pos[:, 0], pos[:, 1], marker=".",
               c=np.arange(0, len(actions)+1), cmap="jet")
    ax.set_xlim([0, XLIM])
    ax.set_ylim([0, YLIM])

    fig, ax = plt.subplots()
    ax.imshow(reward)
    ax.set_xlim([0, XLIM])
    ax.set_ylim([0, YLIM])
