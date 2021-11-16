# -*- coding: utf-8 -*-
"""
@author: Dennis Sand, Nicolai Almskou, Peter Fisker & Victor Nissen
"""

# %% Imports
import os

import numpy as np
from numpy.random import multivariate_normal as mnormal

from tqdm import tqdm

import classes
import helpers

# %% Global Variables

# Enviroment parameters
XLIM = 10
YLIM = 10

# Action space
NBEAMS = 8

# Simulation parameters
NSTEP = 1000
NEPISODES = 100

# Parameters when creating the covariance matrix
RHO = 0.8  # Correlation factor
SIGMA_SQUARED = 100  # Noise variance

# Methods:
POLICY = "UBC"  # "greedy", "e_greedy", "UBC"
UPDATE = "SARSA"  # "simple", "SARSA", "Q_LEARNING"
ALPHA = ["constant", 0.7]  # ["method", "start_value"] - "constant", "1/n"

# PLOTS
PLOT = True

# %% main

if __name__ == '__main__':
    print("Starting", flush=True)

    # Step space
    step_space = ["left", "right", "up", "down",
                  "left-up", "left-down", "right-up", "right-down"]

    # Beam Space
    beam_space = range(NBEAMS)

    # See if cov matrix is avaible, if not create it:
    if os.path.exists(f"cov/cov_{XLIM}x{YLIM}_{RHO}_{SIGMA_SQUARED}.npy"):
        print("Load covariance matrix", flush=True)
        cov = np.load(f"cov/cov_{XLIM}x{YLIM}_{RHO}_{SIGMA_SQUARED}.npy")
    else:
        print("Create covariance matrix", flush=True)
        cov = helpers.getCov(XLIM, YLIM, RHO, SIGMA_SQUARED)
        if not os.path.exists("cov"):
            os.makedirs("cov")
        np.save(f"cov/cov_{XLIM}x{YLIM}_{RHO}_{SIGMA_SQUARED}.npy", cov)

    # Create the reward matrices
    print("Create reward matrix", flush=True)
    reward = np.zeros([XLIM, YLIM, len(beam_space)])

    for idx, _ in enumerate(beam_space):
        reward_tmp = mnormal(mean=np.zeros(XLIM * YLIM) + 150, cov=cov, size=1)
        reward[:, :, idx] = reward_tmp.reshape([XLIM, YLIM])

    # Get environment
    print("Creating enviroment", flush=True)
    env = classes.GridWorld(XLIM, YLIM, reward, sigma=1)

    # Create agent.
    print('Creating agents', flush=True)
    if POLICY == "UBC":
        agents = [classes.Agent(action_space=beam_space, alpha=ALPHA, gamma=0.7, c=x)
                  for x in [100, 1000, 10000, 100000]]
    else:
        agents = [classes.Agent(action_space=beam_space, alpha=ALPHA, gamma=0.7, eps=x)
                  for x in np.linspace(0.0, 0.01, 2, endpoint=True)]

    print('Start simulating', flush=True)
    for episode in tqdm(range(NEPISODES), desc="Episodes:"):
        for agent in agents:
            helpers.game(env, agent, step_space, NSTEP, POLICY, [XLIM, YLIM], UPDATE, episode)

    print("\nResults:", flush=True)
    for agent in agents:
        print(f"We choose the optimal choice {agent.accuracy[-1] * 100:.2f}%" +
              f" of the time with {agent.eps:.2f}-greedy policy")

    # %% plots
    if PLOT:
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        # plots the tru mean of the reward matrices for each beam direction.
        if NBEAMS < 8:
            vmin = min(np.min(reward), np.min(reward))
            vmax = max(np.max(reward), np.max(reward))

            fig, ax = plt.subplots(np.shape(reward)[-1], 1, sharex=True)
            for idx in range(np.shape(reward)[-1]):
                ax[idx].set_title(f"Reward {idx} - mean")
                s = ax[idx].imshow(reward[:, :, idx], vmin=vmin, vmax=vmax, origin="lower",
                                   interpolation='None', aspect="auto")
                ax[idx].set_xlim([-1, XLIM])
                ax[idx].set_ylim([-1, YLIM])

            axins = inset_axes(ax[-1],
                               width="100%",  # width = 100% of parent_bbox width
                               height="5%",  # height : 5%
                               loc='lower center',
                               bbox_to_anchor=(0, -0.5, 1, 1),
                               bbox_transform=ax[-1].transAxes,
                               borderpad=0,
                               )

            fig.colorbar(s, ax=ax[-1], cax=axins, orientation="horizontal")

            fig.set_figheight(15)
            fig.set_figwidth(15)
            fig.tight_layout(pad=3.0)

        # plots the accuracy for each episode
        plt.figure(2)
        if POLICY == "UBC":
            for agent in agents:
                plt.plot(agent.accuracy[1:] * 100, label=f'c={agent.c:.2f}')
        else:
            for agent in agents:
                plt.plot(agent.accuracy[1:] * 100, label=f'Eps={agent.eps:.2f}')

        plt.title(f'{UPDATE} - Steps: {NSTEP}, alpha: {ALPHA}')
        plt.xlabel('Number of episodes')
        plt.ylabel('%-age the optimal choice is chosen [%]')
        plt.legend()
        plt.show()
