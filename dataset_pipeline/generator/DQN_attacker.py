import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .state_action_tools import AbstractAttackerModel

Q_NETWORK_ITERATION = 100

BoolTensor = torch.cuda.BoolTensor if torch.cuda.is_available() else torch.BoolTensor
FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor


class Net(nn.Module):
    def __init__(self, num_state, num_action):
        super().__init__()
        self.fc1 = nn.Linear(num_state, 512)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(512, 512)
        self.fc2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(512, num_action)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.out(x)
        return q_value


class DQN_attacker(AbstractAttackerModel):
    batch_size = 512
    memory_capacity = 10000
    learning_rate = 1e-5
    max_grad_norm = 0.5
    reward_scaling = 0.01
    epsilon = 0.9

    def __init__(self, stateaction_model):
        super().__init__(stateaction_model)
        self.eval_net = Net(num_state=self.state_dim, num_action=self.action_dim)
        self.target_net = Net(num_state=self.state_dim, num_action=self.action_dim)

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory_full = False
        self.gamma = 0.99
        self.memory = np.zeros((self.memory_capacity, self.state_dim * 2 + self.action_dim * 2 + 3))

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.learning_rate)
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.eval_net.cuda()
            self.target_net.cuda()

        self.loss_func = nn.MSELoss()

    def save_param(self, path_str, i_episode):
        torch.save(self.eval_net.state_dict(), os.path.join(os.path.join("./Model", path_str), f"attacker-{i_episode}.pkl"))

    def choose_abstract_action(self, current_state, action_mask, test_flag=False):
        current_state = torch.from_numpy(current_state).float().unsqueeze(0)
        if not self.memory_full:
            valid_action_index = [i for i in range(self.action_dim) if action_mask[i]]
            return np.random.choice(valid_action_index), None

        if np.random.randn() <= self.epsilon:
            action_value = self.eval_net(current_state)
            mask_action_value = torch.masked_select(action_value, torch.tensor(action_mask).type(BoolTensor)).view(1, -1)
            pick_valid_action_index = torch.max(mask_action_value, 1)[1].data.cpu().numpy()
            valid_action_list = [i for i in range(self.action_dim) if action_mask[i]]
            action = valid_action_list[pick_valid_action_index[0]]
        else:
            valid_actions = [i for i in range(self.action_dim) if action_mask[i]]
            action = np.random.choice(valid_actions)
        return action, None

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

        if self.learn_step_counter % Q_NETWORK_ITERATION == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(self.memory_capacity, self.batch_size)
        batch_memory = self.memory[sample_index, :]

        state = torch.tensor(batch_memory[:, :self.state_dim], dtype=torch.float).type(FloatTensor)
        action = torch.tensor(batch_memory[:, self.state_dim:self.state_dim + 1], dtype=torch.long).type(LongTensor)
        reward = torch.tensor(batch_memory[:, self.state_dim + 1:self.state_dim + 2], dtype=torch.float).type(FloatTensor)
        next_state = torch.tensor(batch_memory[:, self.state_dim + 2:2 * self.state_dim + 2], dtype=torch.float).type(FloatTensor)
        mask = torch.tensor(batch_memory[:, 2 * self.state_dim + 2:2 * self.state_dim + 3], dtype=torch.float).type(FloatTensor)

        q_eval = self.eval_net(state).gather(1, action)
        q_next = self.target_net(next_state).detach()
        q_target = reward.view(self.batch_size, 1) + self.gamma * (1 - mask).view(self.batch_size, 1) * q_next.max(1)[0].view(self.batch_size, 1)
        loss = self.loss_func(q_eval, q_target)

        if path_str:
            loss_file = open(os.path.join(os.path.join("./Outcome", path_str), "loss.txt"), "a+")
            loss_file.write(f"{float(loss)}\n")
            loss_file.close()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.eval_net.parameters(), self.max_grad_norm)
        self.optimizer.step()
