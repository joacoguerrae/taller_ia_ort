import numpy as np
import matplotlib.pyplot as plt


def plot_rewards(rewards, average_range=1000):
    episode_ticks = int(len(rewards) / average_range)

    avg_rewards = np.array(rewards).reshape((episode_ticks, average_range))
    avg_rewards = np.mean(avg_rewards, axis=1)

    plt.plot([i * average_range for i in range(episode_ticks)], avg_rewards)
    # plt.plot(rewards)
    plt.title("Episode Accumulated Reward")
    plt.xlabel("Episode Number")
    plt.ylabel("Reward")
    plt.show()


def plot_epsilon(epsilons):
    plt.plot(epsilons)
    plt.title("Epsilon over Episodes")
    plt.xlabel("Episode Number")
    plt.ylabel("Epsilon")
    plt.show()


def test_utils():
    print("This is a test function in Utils.py")
    pass
