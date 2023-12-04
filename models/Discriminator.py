import torch
import random, math
from torch import nn
import numpy as np
import torch.nn.functional as F


class DiscriminatorNN(nn.Module):

    def __init__(self, sa_dim, device="cuda:0"):
        super(DiscriminatorNN, self).__init__()
        self.device = torch.device(device)
        self.hidden1 = nn.Linear(sa_dim, 256).to(device)
        self.hidden2 = nn.Linear(256, 128).to(device)
        self.output = nn.Linear(128, 1).to(device)

    def forward(self, x):
        x = torch.tanh(self.hidden1(x))
        x = torch.tanh(self.hidden2(x))
        return self.output(x)


class Discriminator():

    def __init__(self, state_dim, action_dim, lr, epoch, device="cuda:0"):
        self.disc = DiscriminatorNN(state_dim + action_dim, device=device)
        self.disc_optim = torch.optim.Adam(self.disc.parameters(), lr=lr)
        self.gail_epoch = epoch
        self.batch_size = 256
        self.device = torch.device(device)

    def update(self, replay_buffer_real, replay_buffer_sac):
        loss = 0.0
        sum_expert_loss = 0.0
        sum_policy_loss = 0.0

        expert_acc = []
        policy_acc = []
        for _ in range(self.gail_epoch):
            # expert_batch = random.sample(expert_batch, self.batch_size)
            # policy_batch = random.sample(policy_batch, self.batch_size)
            expert_batch = replay_buffer_real.sample(self.batch_size)
            policy_batch = replay_buffer_sac.sample(self.batch_size)
            # expert_states, expert_actions, _, _, _ = map(torch.FloatTensor, map(np.stack, zip(*expert_batch)))
            # policy_states, policy_actions, _, _, _ = map(torch.FloatTensor, map(np.stack, zip(*policy_batch)))
            expert_states, expert_actions = expert_batch['observations'], expert_batch['actions']
            policy_states, policy_actions = policy_batch['observations'], policy_batch['actions']

            expert_d = self.disc(torch.cat([expert_states, expert_actions], dim=1))
            policy_d = self.disc(torch.cat([policy_states, policy_actions], dim=1))

            expert_acc.append((expert_d > 0).float().mean().item())
            policy_acc.append((policy_d < 0).float().mean().item())

            expert_loss = F.binary_cross_entropy_with_logits(expert_d, torch.ones(expert_d.size(), device=self.device))
            policy_loss = F.binary_cross_entropy_with_logits(policy_d, torch.zeros(policy_d.size(), device=self.device))

            gail_loss = expert_loss + policy_loss

            self.disc_optim.zero_grad()
            gail_loss.backward()
            self.disc_optim.step()

            loss += gail_loss.item()
            sum_expert_loss += expert_loss.item()
            sum_policy_loss += policy_loss.item()

        return loss, np.mean(expert_acc), np.mean(policy_acc)

    def predict_rewards(self, rollout):
        states, actions, logprob_base, rewards, dones = map(np.stack, zip(*rollout))
        with torch.no_grad():
            policy_mix = torch.cat([torch.FloatTensor(states), torch.FloatTensor(actions)], dim=1)
            policy_d = self.disc(policy_mix).squeeze()
            score = torch.sigmoid(policy_d)
            # gail_rewards = - (1-score).log()
            gail_rewards = score.log() - (1 - score).log()
            return (states, actions, logprob_base, gail_rewards.numpy(), dones)

