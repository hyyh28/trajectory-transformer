import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import d4rl
import copy
import numpy as np


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, device, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = device

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

    def save(self, save_folder):
        np.save(f"{save_folder}_state.npy", self.state[:self.size])
        np.save(f"{save_folder}_action.npy", self.action[:self.size])
        np.save(f"{save_folder}_next_state.npy", self.next_state[:self.size])
        np.save(f"{save_folder}_reward.npy", self.reward[:self.size])
        np.save(f"{save_folder}_not_done.npy", self.not_done[:self.size])
        np.save(f"{save_folder}_ptr.npy", self.ptr)

    def load(self, save_folder, size=-1):
        reward_buffer = np.load(f"{save_folder}_reward.npy")

        # Adjust crt_size if we're using a custom size
        size = min(int(size), self.max_size) if size > 0 else self.max_size
        self.size = min(reward_buffer.shape[0], size)

        self.state[:self.size] = np.load(f"{save_folder}_state.npy")[:self.size]
        self.action[:self.size] = np.load(f"{save_folder}_action.npy")[:self.size]
        self.next_state[:self.size] = np.load(f"{save_folder}_next_state.npy")[:self.size]
        self.reward[:self.size] = reward_buffer[:self.size]
        self.not_done[:self.size] = np.load(f"{save_folder}_not_done.npy")[:self.size]


class Potential(nn.Module):
    def __init__(self, state_dim):
        super(Potential, self).__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, state):
        q1 = F.relu(self.l1(state))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class IRS(object):
    def __init__(self, state_dim, device, discount=0.9):
        self.potential = Potential(state_dim).to(device)
        self.potential_target = copy.deepcopy(self.potential)
        self.potential_optimizer = torch.optim.Adam(self.potential.parameters(), lr=1e-3)
        self.discount = discount
        self.device = device

    def train(self, replay_buffer, iterations, batch_size=100):
        for it in range(iterations):
            state, next_state, reward, not_done = replay_buffer.sample(batch_size)
            with torch.no_grad():
                target_q = self.potential_target(next_state)
                target_q = reward + not_done * self.discount * target_q
            current_q = self.potential(state)
            potential_loss = F.mse_loss(current_q, target_q)
            self.potential_optimizer.zero_grad()
            potential_loss.backward()
            self.potential_optimizer.step()


def train_IRS(state_dim, action_dim, device, output_dir, args, discount=0.9):
    reward_shaping = IRS(state_dim, device, discount)
    replay_buffer = ReplayBuffer(state_dim, action_dim, device)
    dataset = get_expert_dataset()
    N = dataset['rewards'].shape[0]
    print('Loading Buffer!')
    for i in range(N-1):
        obs = dataset['observations'][i]
        new_obs = dataset['observations'][i+1]
        action = dataset['actions'][i]
        reward = dataset['rewards'][i]
        done_bool = bool(dataset['terminals'][i])
        replay_buffer.add(obs, action, new_obs, reward, done_bool)
    print('Loaded buffer')
    evaluations = []
    episode_num = 0
    done = True
    training_iters = 0

    while training_iters < args.max_timesteps:
        print("Train step: ", training_iters)
        reward_shaping.train(replay_buffer, iterations=int(args.eval_freq), batch_size=args.batch_size)


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
    return expert_dataset
