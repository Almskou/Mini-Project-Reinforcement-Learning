# -*- coding: utf-8 -*-
"""
@author: Dennis Sand, Nicolai Almskou, Peter Fisker & Victor Nissen
"""

import numba
import numpy as np


@numba.jit
def getCov(Lx, Ly, rho, sigma_squared=1):
    """
    Calculates the covariance matrix based on a correlation factor and
    the distiance between two points in the grid

    Parameters
    ----------
    Lx : INT
        Size of the grids x-axis.
    Ly : INT
        Size of the grids y-axis..
    rho : FLOAT
        Correlation factor [0,1[.
    sigma_squared : FLOAT
        Variance of the covariance matrix.

    Returns
    -------
    cov : FLOAT MATRIX
        Return the covariance based on the set correlation factor.
        Has dimension (Lx*Ly, Lx*Ly)

    """

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

    Parameters
    ----------
    reward : MATRIX
        The reward matrix.
    Q : DICT
        The Q table.
    beam_space : ARRAY
        The possible beam directions.
    limits : ARRAY
        The limit for the x-axis and y-axis (XLIM, YLIM).

    Returns
    -------
    result : MATRIX
        A matrix containing true and false statements for each index in Q.
    optimal_choice : MATRIX
        The matrix which contains the optimal choices.
    Q_choice : MATRIX
        The matrix which contain the choices for the given Q table.

    """

    xlim, ylim = limits

    optimal_choice = np.zeros_like(reward[:, :, 0], dtype=float)
    Q_choice = np.zeros_like(reward[:, :, 0], dtype=float)

    # iterate over all states
    for idx in range(xlim):
        for idy in range(ylim):

            # Find the optimal choice for given state
            optimal_choice[idx, idy] = np.argmax(reward[idx, idy, :])

            # Find the choice for the Q-table
            beam = beam_space[0]
            q = Q[(idx, idy), beam]

            for beam_new in beam_space:
                if Q[(idx, idy), beam_new] >= q:
                    q = Q[(idx, idy), beam_new]
                    beam = beam_new

            Q_choice[idx, idy] = beam

    # Compare
    result = optimal_choice == Q_choice

    return result, optimal_choice, Q_choice


def game(env, agent, step_space, n_step, policy, limits, update, episode):
    """
    Run one episode of the "game"

    Parameters
    ----------
    env : CLASS
        The created GridWorld enviroment.
    agent : CLASS
        The created agent.
    step_space : ARRAY
        The possible steps which can be taken.
    n_step : INT
        Number of steps it should take.
    policy : STR
        Choose a politcy (greedy / e_greedy).
    limits : ARRAY
        The limit for the x-axis and y-axis (XLIM, YLIM).
    update : STR
        Choose the algorithm to update the Q table (simple / SARSA / Q_LEARNING).
    episode: INT
        Which episiode it is currently. Used when policty is "UCB"

    Returns
    -------
    None.

    """

    xlim, ylim = limits

    # Get a random start state
    state = np.random.randint(0, [xlim - 1, ylim - 1], dtype=int)

    # Get an action based on the policy
    if policy == 'e_greedy':
        action = agent.e_greedy(state)
    elif policy == "UCB":
        action = agent.UCB(state, 1 + n_step*episode)
    else:
        action = agent.greedy(state)

    # Create the random walk.
    steps = np.random.choice(step_space, n_step, replace=True)

    # Walk the the random walk
    for idx, step in enumerate(steps):
        # Get the next step based on taken action and current state
        next_state, R = env.step(state, step, action)

        # Get the next action based on chosen policy
        if policy == 'e_greedy':
            next_action = agent.e_greedy(next_state)
        elif policy == "UCB":
            next_action = agent.UCB(next_state, 1 + idx + n_step*episode)
        else:
            next_action = agent.greedy(next_state)

        # Update the Q table
        if update == "SARSA":
            agent.update_sarsa(R, state, action, next_state, next_action)
        elif update == "Q_LEARNING":
            agent.update_Q_learning(R, state, action, next_state)
        else:
            agent.update(state, action, R)

        state = next_state
        action = next_action

    # Compare the greedy policy on the created Q table with
    # the greedy policy on the true mean reward functions
    result = check_result(env.reward, agent.Q, agent.action_space, limits)
    accuracy = np.count_nonzero(result[0]) / result[0].size

    # Save the accuracy
    agent.accuracy = np.append(agent.accuracy, accuracy)
