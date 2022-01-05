import numpy as np
import pickle
expert_file = 'maze_expert.npy'
imitation_agent_file = 'maze_agent.npy'
with open(imitation_agent_file, 'rb') as handle:
    agent_data = pickle.load(handle)
with open(expert_file, 'rb') as handle:
    expert_data = pickle.load(handle)
print("OK")