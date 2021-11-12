# -*- coding: utf-8 -*-
"""
@author: Dennis Sand, Nicolai Almskou, Peter Fisker & Victor Nissen
"""

# %% Imports
import os

import numpy as np
from numpy.random import multivariate_normal as mnormal

import classes
import helpers

# %% Global Variables
XLIM = 30
YLIM = 30
NSTEP = 1000
NEPISODES = 1000
RHO = 0.8
NBEAMS = 4
PLOT = True

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
        cov = helpers.getCov(XLIM, YLIM, rho)
        if not os.path.exists("cov"):
            os.makedirs("cov")
        np.save(f"cov/cov_{XLIM}x{YLIM}_{rho}.npy", cov)

    # For this example we have two beam direction 0 deg and 180 deg.
    # Therefor we need to create a reward matrix for each beam direction.
    # Reward matrix for 0 deg.
    reward = np.zeros([XLIM, YLIM, len(beam_space)])

    for idx, _ in enumerate(beam_space):
        reward_tmp = mnormal(mean=np.zeros(XLIM * YLIM) + 150, cov=cov, size=1)
        # reward_tmp = 30*mnormal(mean=np.zeros(XLIM * YLIM), cov=cov, size=1)
        reward[:, :, idx] = reward_tmp.reshape([XLIM, YLIM])

    print('Creating agents')
    # Create agent.

    # Get environment
    env = classes.GridWorld(XLIM, YLIM, reward, sigma=1)

    percent_greedy = np.zeros(NEPISODES)
    percent_e_greedy = np.zeros(NEPISODES)
    agent_greedy = classes.Agent(action_space=beam_space)
    agent_e_greedy = classes.Agent(action_space=beam_space)

    for episode in range(NEPISODES):
        if not (episode % 10):
            print(episode)
        # agent_greedy = classes.Agent(action_space=beam_space)
        # agent_e_greedy = classes.Agent(action_space=beam_space)

        greedy_game = helpers.game(env, agent_greedy,
                                   step_space, NSTEP, 'greedy',
                                   [XLIM, YLIM])
        percent_greedy[episode] = np.count_nonzero(greedy_game[0]) / greedy_game[0].size

        e_greedy_game = helpers.game(env, agent_e_greedy,
                                     step_space, NSTEP, 'e_greedy',
                                     [XLIM, YLIM])
        percent_e_greedy[episode] = np.count_nonzero(e_greedy_game[0]) / e_greedy_game[0].size

    print(f'We choose the optimal choice {(percent_greedy[-1]) * 100}% of the time with greedy policy')

    print(f'We choose the optimal choice {(percent_e_greedy[-1]) * 100}% of the time with e_greedy policy')

    # %% plots
    if PLOT:
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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

        plt.figure(2)
        plt.plot(percent_greedy)
        plt.plot(percent_e_greedy)
        plt.show()
