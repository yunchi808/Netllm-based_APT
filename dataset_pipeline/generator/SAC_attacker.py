import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from .state_action_tools import AbstractAttackerModel


BoolTensor = torch.cuda.BoolTensor if torch.cuda.is_available() else torch.BoolTensor
FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor


class Actor(nn.Module):
    def __init__(self, num_state, num_action):
        super().__init__()
        self.fc1 = nn.Linear(num_state, 256)
        self.fc2 = nn.Linear(256, 512)
        self.action_head = nn.Linear(512, num_action)

    def forward(self, x, action_mask):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.action_head(x)
        masks = action_mask.type(BoolTensor)
        logits = torch.where(masks, x, torch.tensor(-1e8).type(FloatTensor))
        action_prob = F.softmax(logits, dim=1)
        return action_prob


class Critic(nn.Module):
    def __init__(self, num_state, num_action):
        super().__init__()
        self.fc1 = nn.Linear(num_state, 512)
        self.fc2 = nn.Linear(512, 512)
        self.state_value = nn.Linear(512, num_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.state_value(x)
        return value


class SAC_attacker(AbstractAttackerModel):
    max_grad_norm = 0.001
    batch_size = 512
    soft_update_tau = 0.001
    memory_capacity = 100000
    learning_rate = 1e-5
    reward_scaling = 1

    def __init__(self, stateaction_model):
        super().__init__(stateaction_model)
        self.memory = np.zeros((self.memory_capacity, self.state_dim * 2 + self.action_dim * 2 + 3))
        self.memory_counter = 0
        self.memory_full = False
        self.gamma = 0.99

        self.target_entropy = -self.action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optim = optim.Adam([self.log_alpha], lr=self.learning_rate)

        self.attacker_net = Actor(num_state=self.state_dim, num_action=self.action_dim)
        self.critic_local = Critic(num_state=self.state_dim, num_action=self.action_dim)
        self.critic_local2 = Critic(num_state=self.state_dim, num_action=self.action_dim)
        self.critic_target = Critic(num_state=self.state_dim, num_action=self.action_dim)
        self.critic_target2 = Critic(num_state=self.state_dim, num_action=self.action_dim)

        self.critic_target.load_state_dict(self.critic_local.state_dict())
        self.critic_target2.load_state_dict(self.critic_local2.state_dict())

        self.actor_optimizer = optim.Adam(self.attacker_net.parameters(), lr=self.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=5 * self.learning_rate)
        self.critic_optimizer2 = optim.Adam(self.critic_local2.parameters(), lr=5 * self.learning_rate)

        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.attacker_net.cuda()
            self.critic_local.cuda()
            self.critic_local2.cuda()
            self.critic_target.cuda()
            self.critic_target2.cuda()

    def save_param(self, path_str, i_episode):
        torch.save(self.attacker_net.state_dict(), os.path.join(os.path.join("./Model", path_str), f"attacker-{i_episode}.pkl"))

    def produce_action_and_action_info(self, state, action_mask=None):
        action_probs = self.attacker_net(state.type(FloatTensor), torch.tensor(action_mask))
        c = Categorical(action_probs)
        action = c.sample()
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probs + z)
        return action, (action_probs, log_action_probabilities)

    def choose_abstract_action(self, current_state, action_mask):
        state = torch.from_numpy(current_state).float().unsqueeze(0)
        if not self.memory_full:
            valid_action_index = [i for i in range(self.action_dim) if action_mask[i]]
            return np.random.choice(valid_action_index), None

        action, prob = self.produce_action_and_action_info(state, action_mask)
        action = action.detach().cpu().numpy()
        prob = prob[1].detach().cpu().numpy()
        return action[0], prob[0]

    def store_transition(self, state, action, reward, next_state, done, action_mask, next_action_mask):
        transition = np.hstack((state, [action, reward], next_state, [1.0 if done else 0.0], action_mask, next_action_mask))
        self.memory[self.memory_counter, :] = transition
        self.memory_counter = (self.memory_counter + 1) % self.memory_capacity
        if self.memory_counter == 0:
            self.memory_full = True

    def learn(self, state, action, reward, next_state, path_str, done, a_prob, action_mask, next_action_mask):
        self.store_transition(state, action, reward * self.reward_scaling, next_state, done, action_mask, next_action_mask)
        if not self.memory_full:
            return

        sample_index = np.random.choice(self.memory_capacity, self.batch_size)
        batch_memory = self.memory[sample_index, :]

        state_batch = torch.tensor(batch_memory[:, :self.state_dim], dtype=torch.float).type(FloatTensor)
        action_batch = torch.tensor(batch_memory[:, self.state_dim:self.state_dim + 1], dtype=torch.long).type(LongTensor)
        reward_batch = torch.tensor(batch_memory[:, self.state_dim + 1:self.state_dim + 2], dtype=torch.float).type(FloatTensor)
        next_state_batch = torch.tensor(batch_memory[:, self.state_dim + 2:2 * self.state_dim + 2], dtype=torch.float).type(FloatTensor)
        mask_batch = torch.tensor(batch_memory[:, 2 * self.state_dim + 2:2 * self.state_dim + 3], dtype=torch.float).type(FloatTensor)
        action_mask_batch = torch.tensor(
            batch_memory[:, 2 * self.state_dim + 3:2 * self.state_dim + 3 + self.action_dim], dtype=torch.bool
        ).type(BoolTensor)
        next_action_mask_batch = torch.tensor(
            batch_memory[:, 2 * self.state_dim + 3 + self.action_dim:2 * self.state_dim + 3 + 2 * self.action_dim], dtype=torch.bool
        ).type(BoolTensor)

        policy_loss, log_pis = self.calculate_actor_loss(state_batch, action_mask_batch)
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.attacker_net.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        alpha_loss = -((self.log_alpha.type(FloatTensor)).exp() * (log_pis + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp().detach()

        with torch.no_grad():
            _, (action_probabilities, log_action_probabilities) = self.produce_action_and_action_info(next_state_batch, next_action_mask_batch)
            qf1_next_target = self.critic_target(next_state_batch)
            qf2_next_target = self.critic_target2(next_state_batch)
            min_qf_next_target = action_probabilities * (
                torch.min(qf1_next_target, qf2_next_target) - self.alpha.type(FloatTensor) * log_action_probabilities
            )
            min_qf_next_target = min_qf_next_target.sum(dim=1).unsqueeze(-1)
            next_q_value = reward_batch + (1.0 - mask_batch) * self.gamma * min_qf_next_target

        qf1 = self.critic_local(state_batch).gather(1, action_batch.long())
        qf2 = self.critic_local2(state_batch).gather(1, action_batch.long())
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)

        self.critic_optimizer.zero_grad()
        qf1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        self.critic_optimizer2.zero_grad()
        qf2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local2.parameters(), self.max_grad_norm)
        self.critic_optimizer2.step()

        self.soft_update_of_target_network(self.critic_local, self.critic_target, self.soft_update_tau)
        self.soft_update_of_target_network(self.critic_local2, self.critic_target2, self.soft_update_tau)

        if path_str:
            aloss_file = open(os.path.join(os.path.join("./Outcome", path_str), "a_loss.txt"), "a+")
            closs_file = open(os.path.join(os.path.join("./Outcome", path_str), "c_loss.txt"), "a+")
            alphaloss_file = open(os.path.join(os.path.join("./Outcome", path_str), "alpha_loss.txt"), "a+")
            aloss_file.write(f"{float(policy_loss)}\n")
            closs_file.write(f"{float(qf1_loss)}\n")
            alphaloss_file.write(f"{float(alpha_loss)}\n")
            aloss_file.close()
            closs_file.close()
            alphaloss_file.close()

    def calculate_actor_loss(self, state_batch, action_mask_batch):
        action, (action_probabilities, log_action_probabilities) = self.produce_action_and_action_info(state_batch, action_mask_batch)
        qf1_pi = self.critic_local(state_batch)
        qf2_pi = self.critic_local2(state_batch)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        inside_term = self.alpha.type(FloatTensor) * log_action_probabilities - min_qf_pi
        policy_loss = (action_probabilities * inside_term).sum(dim=1).mean()
        log_action_probabilities = torch.sum(log_action_probabilities * action_probabilities, dim=1)
        return policy_loss, log_action_probabilities

    def soft_update_of_target_network(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
