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
XLIM = 15
YLIM = 15
NSTEP = 100
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

    reward = np.zeros([XLIM, YLIM, len(beam_space)])

    for idx, _ in enumerate(beam_space):
        reward_tmp = mnormal(mean=np.zeros(XLIM * YLIM) + 150, cov=cov*1000, size=1)
        reward[:, :, idx] = reward_tmp.reshape([XLIM, YLIM])

    # Get environment
    env = classes.GridWorld(XLIM, YLIM, reward, sigma=1)

    # Create agent.
    print('Creating agents')
    agents = [classes.Agent(action_space=beam_space, gamma=0.7, eps=x) for x in np.linspace(0.0, 0.01, 2, endpoint=True)]

    for episode in range(NEPISODES):
        if not (episode % 10):
            print(episode)

        for agent in agents:
            helpers.game(env, agent, step_space, NSTEP, 'e_greedy', [XLIM, YLIM])

    for agent in agents:
        print(
            f'We choose the optimal choice {agent.accuracy[-1] * 100:.2f}% of the time with {agent.eps:.2f}-greedy policy')

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
        for agent in agents:
            plt.plot(agent.accuracy[1:] * 100, label=f'Eps={agent.eps:.2f}')

        plt.title('Epsilon-greedy with varying epsilon')
        plt.xlabel('Number of episodes')
        plt.ylabel('%-age the optimal choice is chosen [%]')
        plt.legend()
        plt.show()
