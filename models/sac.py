from ml_collections import ConfigDict

import torch
import torch.nn.functional as F

from models.model import Scalar, soft_target_update


class SAC(object):

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.discount = 0.99
        config.reward_scale = 1.0
        config.alpha_multiplier = 1.0
        config.use_automatic_entropy_tuning = True
        config.backup_entropy = True
        config.target_entropy = 0.0
        config.policy_lr = 3e-4
        config.qf_lr = 3e-4
        config.optimizer_type = 'adam'
        config.soft_target_update_rate = 5e-3
        config.target_update_period = 1
        config.alpha_l1 = 1.0

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, policy, qf1, qf2, target_qf1, target_qf2, deviation_theta=0, alpha_l1=1):
        self.config = SAC.get_default_config(config)
        self.config.alpha_l1 = alpha_l1
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.deviation_theta = torch.tensor(deviation_theta * torch.pi / 180, dtype=torch.float64)

        optimizer_class = {
            'adam': torch.optim.Adam,
            'sgd': torch.optim.SGD,
        }[self.config.optimizer_type]

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(), self.config.policy_lr,
        )
        self.qf_optimizer = optimizer_class(
            list(self.qf1.parameters()) + list(self.qf2.parameters()), self.config.qf_lr
        )
        # self.l1_optimizer = optimizer_class(
        #     list(self.qf1.embedding_network.parameters()) +
        #     list(self.qf2.embedding_network.parameters()), self.config.qf_lr
        # )

        if self.config.use_automatic_entropy_tuning:
            self.log_alpha = Scalar(0.0)
            self.alpha_optimizer = optimizer_class(
                self.log_alpha.parameters(),
                lr=self.config.policy_lr,
            )
        else:
            self.log_alpha = None
        self.alpha = self.log_alpha().detach().exp() * self.config.alpha_multiplier

        self.update_target_network(1.0)
        self._total_steps = 0

    def update_target_network(self, soft_target_update_rate):
        soft_target_update(self.qf1, self.target_qf1, soft_target_update_rate)
        soft_target_update(self.qf2, self.target_qf2, soft_target_update_rate)

    def train(self, batch, batch_real):
        self._total_steps += 1

        observations = batch['observations']
        actions = batch['actions']
        rewards = batch['rewards']
        next_observations = batch['next_observations']
        dones = batch['dones']

        """ Q function loss """
        q1_pred = self.qf1(observations, actions)
        q2_pred = self.qf2(observations, actions)

        with torch.no_grad():
            new_next_actions, next_log_pi = self.policy(next_observations)
            target_q_values = torch.min(
                self.target_qf1(next_observations, new_next_actions),
                self.target_qf2(next_observations, new_next_actions),
            )
            if self.config.backup_entropy:
                target_q_values = target_q_values - self.alpha * next_log_pi
            q_target = self.config.reward_scale * torch.squeeze(rewards, -1) + (1. - torch.squeeze(dones, -1)) * self.config.discount * target_q_values

        qf1_loss = F.mse_loss(q1_pred, q_target)
        qf2_loss = F.mse_loss(q2_pred, q_target)
        qf_loss = qf1_loss + qf2_loss

        # """ L1 loss """
        # with torch.no_grad():
        real_observations = batch_real['observations']
        real_actions = batch_real['actions']
        pi_actions = self.policy(torch.zeros(observations.shape, device='cuda:0'))[0]
        qf1_kernal = torch.sum(self.qf1.embedding(real_observations, real_actions) *
                               self.qf1.embedding(real_observations, pi_actions), dim=1)
        qf2_kernal = torch.sum(self.qf2.embedding(real_observations, real_actions) *
                               self.qf2.embedding(real_observations, pi_actions), dim=1)
        qf1_l1_loss = torch.abs(qf1_kernal - torch.cos(self.deviation_theta)).mean()
        qf2_l1_loss = torch.abs(qf2_kernal - torch.cos(self.deviation_theta)).mean()
        l1_loss = self.config.alpha_l1 * (qf1_l1_loss + qf2_l1_loss)

        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        l1_loss.backward()
        self.qf_optimizer.step()

        """ Policy loss """
        new_actions, log_pi = self.policy(observations)
        q_new_actions = torch.min(
            self.qf1(observations, new_actions),
            self.qf2(observations, new_actions),
        )
        policy_loss = (self.alpha * log_pi - q_new_actions).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        if self.config.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha() * (log_pi + self.config.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha().detach().exp() * self.config.alpha_multiplier
            # pdb.set_trace()
        else:
            alpha_loss = observations.new_tensor(0.0)
            self.alpha = observations.new_tensor(self.config.alpha_multiplier)


        if self.total_steps % self.config.target_update_period == 0:
            self.update_target_network(
                self.config.soft_target_update_rate
            )

        return dict(
            log_pi=log_pi.mean().item(),
            policy_loss=policy_loss.item(),
            qf1_loss=qf1_loss.item(),
            qf2_loss=qf2_loss.item(),
            alpha_loss=alpha_loss.item(),
            alpha=self.alpha.item(),
            average_qf1=q1_pred.mean().item(),
            average_qf2=q2_pred.mean().item(),
            average_target_q=target_q_values.mean().item(),
            total_steps=self.total_steps,
        )

    def torch_to_device(self, device):
        for module in self.modules:
            module.to(device)

    @property
    def modules(self):
        modules = [self.policy, self.qf1, self.qf2, self.target_qf1, self.target_qf2]
        if self.config.use_automatic_entropy_tuning:
            modules.append(self.log_alpha)
        return modules

    @property
    def total_steps(self):
        return self._total_steps
