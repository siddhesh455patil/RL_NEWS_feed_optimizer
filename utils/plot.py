import matplotlib.pyplot as plt
import numpy as np

def plot_rewards(rewards):

    moving_avg = np.convolve(rewards, np.ones(500)/500, mode='valid')
    plt.plot(moving_avg)
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.title("Learning Curve (Smoothed)")
    plt.show()