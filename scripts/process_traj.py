import gym
import d4rl
import h5py
import numpy as np
import pickle


env = gym.make("maze2d-umaze-v1")
dataset = env.get_dataset()
expert_data = {'obs_traj': [], 'reward_traj': [], 'target_traj': []}
agent_data = {'obs_traj': [], 'reward_traj': [], 'target_traj': []}

expert_traj = []
expert_traj_label = []
expert_traj_target = []
agent_traj = []
agent_traj_label = []
agent_traj_target = []
for i in range(len(dataset['rewards'])):
    if i % 2 == 0 or (dataset['rewards'][i] > 0 or dataset['timeouts'][i] is True):
        expert_traj.append(dataset['observations'][i][0:2])
        expert_traj_label.append(dataset['rewards'][i])
        expert_traj_target.append(dataset['infos/goal'][i])
    agent_traj.append(dataset['observations'][i])
    agent_traj_label.append(dataset['rewards'][i])
    agent_traj_target.append(dataset['infos/goal'][i])
    if dataset['rewards'][i] > 0 or dataset['timeouts'][i] is True:
        if len(expert_traj) > 2:
            expert_data['obs_traj'].append(np.array(expert_traj))
            expert_data['reward_traj'].append(np.array(expert_traj_label))
            expert_data['target_traj'].append(np.array(expert_traj_target))
            agent_data['obs_traj'].append(np.array(agent_traj))
            agent_data['reward_traj'].append(np.array(agent_traj_label))
            agent_data['target_traj'].append(np.array(agent_traj_target))
        expert_traj = []
        expert_traj_label = []
        expert_traj_target = []
        agent_traj = []
        agent_traj_label = []
        agent_traj_target = []
    if i % 100000 == 0:
        print(i)

expert_file = 'maze_expert.npy'
imitation_agent_file = 'maze_agent.npy'
with open(expert_file, 'wb') as handle:
    pickle.dump(expert_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(imitation_agent_file, 'wb') as handle:
    pickle.dump(agent_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("OK")