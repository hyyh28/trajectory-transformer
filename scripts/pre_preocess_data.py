import gym
import d4rl
import h5py
import numpy as np

env = gym.make("maze2d-umaze-v1")
dataset = env.get_dataset()
count_trajectories = 0
expert_dataset = {}
imitation_agent_dataset = {}
for key in dataset.keys():
    expert_dataset[key] = []
    imitation_agent_dataset[key] = []
for i in range(len(dataset['rewards'])):
    if dataset['rewards'][i] > 0 or dataset['timeouts'][i] is True:
        count_trajectories += 1
        # print(dataset['observations'][i])
print(count_trajectories)

for i in range(len(dataset['rewards'])):
    if i % 2 == 0 or ((i + 1) < len(dataset['rewards']) and (dataset['rewards'][i + 1] > 0
                      or dataset['timeouts'][i + 1] is True)):
        for key in dataset.keys():
            if key != 'observations':
                expert_dataset[key].append(dataset[key][i])
            else:
                expert_dataset[key].append(dataset[key][i][0:2])
    for key in dataset.keys():
        imitation_agent_dataset[key].append(dataset[key][i])

for key in dataset.keys():
    expert_dataset[key] = np.array(expert_dataset[key])
    imitation_agent_dataset[key] = np.array(imitation_agent_dataset[key])
print("Hello")
expert_file = 'maze_expert.npy'
imitation_agent_file = 'maze_agent.npy'
np.save(expert_file, expert_dataset)
np.save(imitation_agent_file, imitation_agent_dataset)
