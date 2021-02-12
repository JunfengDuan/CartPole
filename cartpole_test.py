import gym
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from IPython import display
import torch
from input_extraction import get_screen


# 创建环境并放开step限制（200）
env = gym.make('CartPole-v0').unwrapped

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")






if __name__ == "__main__":
    pass
