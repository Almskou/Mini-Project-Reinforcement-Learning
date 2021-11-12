# -*- coding: utf-8 -*-
"""
@author: Dennis Sand, Nicolai Almskou, Peter Fisker & Victor Nissen
"""

import numba
import numpy as np


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
    sigma_squared = 1000

    pos = np.zeros((Lx * Ly, 2))
    cov = np.zeros((Lx * Ly, Lx * Ly))

    # Get all the position in the grid
    for idx in range(Lx):
        for idy in range(Ly):
            pos[idx * Ly + idy, :] = [idx, idy]

    # Calculate the covariance based on the grid position
    for idx, val in enumerate(pos):
        for idx2, val2 in enumerate(pos):
            cov[idx, idx2] = sigma_squared * np.exp(-np.linalg.norm(val - val2) / (1 / (1 - rho)))

    return cov


def check_result(reward, Q, beam_space, limits):
    """
    Checks if the highest Q-value for each state,
    corresponds to the optimal choice

    :param reward:
    :param Q:
    :return:
    """
    xlim, ylim = limits

    optimal_choice = np.zeros_like(reward[:, :, 0], dtype=float)
    Q_choice = np.zeros_like(reward[:, :, 0], dtype=float)

    for idx in range(xlim):
        for idy in range(ylim):
            optimal_choice[idx, idy] = np.argmax(reward[idx, idy, :])

            beam = beam_space[0]
            q = Q[(idx, idy), beam]

            for beam_new in beam_space:
                if Q[(idx, idy), beam_new] >= q:
                    q = Q[(idx, idy), beam_new]
                    beam = beam_new

            Q_choice[idx, idy] = beam

    result = optimal_choice == Q_choice

    return result, optimal_choice, Q_choice


def game(env, agent, step_space, n_step, policy, limits):
    xlim, ylim = limits

    state = np.random.randint(0, [xlim - 1, ylim - 1], dtype=int)
    if policy == 'e_greedy':
        action = agent.e_greedy(state)
    else:
        action = agent.greedy(state)

    steps = np.random.choice(step_space, n_step, replace=True)

    for step in steps:
        next_state, R = env.step(state, step, action)
        if policy == 'e_greedy':
            next_action = agent.e_greedy(next_state)
        else:
            next_action = agent.greedy(next_state)
        agent.update_sarsa(R, state, action, next_state, next_action)
        # agent.update(state, action, R)
        state = next_state
        action = next_action

    result = check_result(env.reward, agent.Q, agent.action_space, limits)
    accuracy = np.count_nonzero(result[0]) / result[0].size

    agent.accuracy = np.append(agent.accuracy, accuracy)

    return result
