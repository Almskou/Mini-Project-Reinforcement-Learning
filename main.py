# -*- coding: utf-8 -*-
"""
@author: Dennis Sand, Nicolai Almskou, Peter Fisker & Victor Nissen
"""

# %% Imports
import numpy as np
import matplotlib.pyplot as plt

# %% Global Variables
XLIM = 100
YLIM = 100


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
        else:
            raise ValueError

        return next_state


# %%

if __name__ == '__main__':
    env = GridWorld()

    actions = ["up", "left", "left", "up"]

    pos = np.zeros([len(actions)+1, 2])
    pos[0, :] = [50, 50]

    for idx, action in enumerate(actions):
        print(idx)
        pos[idx + 1, :] = env.step(pos[idx, :], action)
