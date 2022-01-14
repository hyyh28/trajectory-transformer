import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import d4rl
import copy
import numpy as np
import pickle
import os
from torch.utils.tensorboard import SummaryWriter
import argparse


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
        self.potential_optimizer = torch.optim.Adam(self.potential.parameters(), lr=5e-5)
        self.discount = discount
        self.device = device

    def train(self, replay_buffer, iterations, batch_size=100):
        global loss_value
        for it in range(iterations):
            state, _, next_state, reward, not_done = replay_buffer.sample(batch_size)
            with torch.no_grad():
                target_q = self.potential_target(next_state)
                target_q = reward + not_done * self.discount * target_q
            current_q = self.potential(state)
            potential_loss = F.mse_loss(current_q, target_q)
            self.potential_optimizer.zero_grad()
            potential_loss.backward()
            self.potential_optimizer.step()
        self.potential_target = copy.deepcopy(self.potential)
        return potential_loss.to('cpu').item()

    def save_model(self, location):
        torch.save(self.potential_target, location)
        print("Model Saved")

    def load_model(self, location):
        self.potential_target = torch.load(location)