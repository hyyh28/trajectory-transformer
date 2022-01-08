import gym
import d4rl
import h5py
import numpy as np


def get_expert_dataset():
    env = gym.make("maze2d-umaze-v1")
    dataset = env.get_dataset()
    expert_dataset = {}
    for key in dataset.keys():
        expert_dataset[key] = []
    for i in range(len(dataset['observations'])):
        if i % 2 == 0 or (dataset['rewards'][i] > 0 or dataset['timeouts'][i] is True):
            for key in dataset.keys():
                expert_dataset[key].append(dataset[key][i])
    for key in dataset.keys():
        expert_dataset[key] = np.array(expert_dataset[key])
    return expert_dataset


dataset = get_expert_dataset()
