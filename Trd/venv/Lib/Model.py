import torch.nn as nn
import torch.optim as optim
import torch

ALPHA = 0.1


class DDQN(nn.Module):
    def __init__(self, obs_len, actions_n, p=0.1, num_envs=64):
        super(DDQN, self).__init__()
        """
        self.fc_adv = nn.Sequential(
            nn.Linear(obs_len, obs_len*2),
            nn.LeakyReLU(ALPHA),
    
            nn.LSTM(input_size = obs_len*2, hidden_size = obs_len*2,num_layers = 1,dropout=0.1),
            nn.LeakyReLU(ALPHA),
    
            nn.Linear(obs_len*2, obs_len),
            nn.LeakyReLU(ALPHA),
    
            nn.LSTM(input_size = obs_len, hidden_size = obs_len,num_layers = 1,dropout=0.1),
            nn.LeakyReLU(ALPHA),
    
            nn.Linear(obs_len, 64),
            nn.ReLU(),
    
            nn.Linear(64, actions_n)
        )
    
        self.fc_val = nn.Sequential(
            nn.Linear(obs_len, obs_len*2),
            nn.LeakyReLU(ALPHA),
    
            nn.LSTM(input_size = obs_len*2, hidden_size = obs_len*2,num_layers = 1,dropout=0.1),
            nn.LeakyReLU(ALPHA),
    
            nn.LSTM(input_size = obs_len*2, hidden_size = obs_len*2,num_layers = 1,dropout=0.1),
            nn.LeakyReLU(ALPHA),
    
            nn.Linear(obs_len*2, obs_len),
            nn.LeakyReLU(ALPHA),
    
            nn.Linear(obs_len, 64),
            nn.ReLU(ALPHA),
    
            nn.Linear(64, 1)
        )"""
        self.length=10
        self.forwards = 0
        self.p = p
        self.num_envs = num_envs
        self.envs=False

        self.hidden_dim_11 = obs_len * 2
        self.hidden_dim_12 = obs_len
        self.num_layers_11 = 1#2
        self.num_layers_12 = 1

        self.hidden_11 = (torch.zeros((self.num_layers_11, 1, self.hidden_dim_11),device="cuda"),
                          torch.zeros((self.num_layers_11, 1, self.hidden_dim_11),device="cuda"))

        self.hidden_12 = (torch.zeros((self.num_layers_12, 1, self.hidden_dim_12),device="cuda"),
                          torch.zeros((self.num_layers_12, 1, self.hidden_dim_12),device="cuda"))

        self.num_layers_21 = 1
        self.hidden_dim_21 = obs_len * 2
        self.hidden_21 = (torch.zeros((self.num_layers_21, 1, self.hidden_dim_21),device="cuda"),
                          torch.zeros((self.num_layers_21, 1, self.hidden_dim_21),device="cuda"))

        self.drop = nn.Dropout(p=self.p)
        self.leak = nn.LeakyReLU(negative_slope=0.2)
        self.relu = nn.ReLU()

        # advances
        self.linear_11 = nn.Linear(obs_len, obs_len * 2)  # 1
        self.linear_12 = nn.Linear(obs_len * 2, 64)#  # 3 obs_len->64
        self.linear_13 = nn.Linear(obs_len, 64)  # 5
        self.linear_14 = nn.Linear(64, actions_n)  # 6

        self.lstm_11 = nn.LSTM(input_size=obs_len * 2, hidden_size=self.hidden_dim_11,
                               num_layers=self.num_layers_11, batch_first=True)  # 2

        self.lstm_12 = nn.LSTM(input_size=obs_len, hidden_size=self.hidden_dim_12,
                               num_layers=self.num_layers_12, batch_first=True)  # 4 #
        # value
        self.linear_21 = nn.Linear(obs_len, obs_len * 2)  # 1

        self.lstm_21 = nn.LSTM(input_size=obs_len * 2, hidden_size=self.hidden_dim_21,
                               num_layers=self.num_layers_21, batch_first=True)

        self.linear_22 = nn.Linear(obs_len * 2, obs_len)  # 3
        self.linear_23 = nn.Linear(2*obs_len, 64)  # 5 #obs_len->*2
        self.linear_24 = nn.Linear(64, 1)

    def rise_length(self):
        self.length+=2

    def reset(self):
        self.forwards=0
        self.hidden_11 = (torch.zeros((self.num_layers_11, 1, self.hidden_dim_11),device="cuda"),
                          torch.zeros((self.num_layers_11, 1, self.hidden_dim_11),device="cuda"))

        self.hidden_12 = (torch.zeros((self.num_layers_12, 1, self.hidden_dim_12),device="cuda"),
                          torch.zeros((self.num_layers_12, 1, self.hidden_dim_12),device="cuda"))

        self.hidden_21 = (torch.zeros((self.num_layers_21, 1, self.hidden_dim_21),device="cuda"),
                          torch.zeros((self.num_layers_21, 1, self.hidden_dim_21),device="cuda"))
        if self.envs:
            self.dict[self.name] = [self.hidden_11, self.hidden_12, self.hidden_21]

    def set_envs(self,envs, name, change):
        self.envs = True
        self.dict = {env.name: [(torch.zeros((self.num_layers_11, 1, self.hidden_dim_11),device="cuda"),
                          torch.zeros((self.num_layers_11, 1, self.hidden_dim_11),device="cuda")),

                                (torch.zeros((self.num_layers_11, 1, self.hidden_dim_11),device="cuda"),
                          torch.zeros((self.num_layers_11, 1, self.hidden_dim_11),device="cuda")),

                                (torch.zeros((self.num_layers_11, 1, self.hidden_dim_11), device="cuda"),
                                 torch.zeros((self.num_layers_11, 1, self.hidden_dim_11), device="cuda"))

                                ] for env in envs}
        self.name  = name
        self.change = change


    def forward(self, var):
        self.forwards+=1
        if(self.forwards==100000):
            self.reset()
        if (len(var.shape) == 2):
            x = var.unsqueeze(1)
        else:
            x = var
        if self.envs and self.change:
            self.hidden_11, self.hidden_12, self.hidden_21 = self.dict[self.name]
        adv = self.linear_11(x)
        adv = self.leak(adv)
        adv, self.hidden_11 = self.lstm_11(adv, self.hidden_11)
        adv = adv.view(-1, self.hidden_dim_11)
        adv = self.leak(adv)
        adv = self.linear_12(adv)
        adv = self.relu(adv)#leak
        """adv = adv.unsqueeze(1)
        adv, self.hidden_12 = self.lstm_12(adv, self.hidden_12)
        adv = adv.view(-1, self.hidden_dim_12)
        adv = self.leak(adv)"""
        adv = self.drop(adv)
        """
        adv = self.linear_13(adv)
        adv = self.relu(adv)
        adv = self.drop(adv)"""
        adv = self.linear_14(adv)

        val = self.linear_21(x)
        val = self.leak(val)
        val, self.hidden_21 = self.lstm_21(val, self.hidden_21)
        val = val.view(-1, self.hidden_dim_21)
        val = self.leak(val)
        """
        val = self.linear_22(val)
        val = self.relu(val)
        val = self.drop(val)"""
        val = self.linear_23(val)
        val = self.relu(val)
        val = self.drop(val)
        val = self.linear_24(val)

        if self.envs:
            self.dict[self.name] = [self.hidden_11, self.hidden_12, self.hidden_21]

        return adv, val


import ptan
import numpy as np
import torch
import torch.nn as nn

HID_SIZE = 64


# Предполагаем, что Actor вычисляет advantages
# Critic вычисляет V(s)
# A2C is just an agent, which returns actions, that he would take in particular state
#

class ModelActor(nn.Module):
    def __init__(self, obs_size, act_size):
        super(ModelActor, self).__init__()

        self.mu = nn.Sequential(
            nn.Linear(obs_size, HID_SIZE),
            nn.Tanh(),
            nn.Linear(HID_SIZE, HID_SIZE),
            nn.Tanh(),
            nn.Linear(HID_SIZE, act_size),
            nn.Tanh()
        )

        self.logstd = nn.Parameter(torch.zeros(
            act_size))  # some sort of parameter in A2c down below with it's help we add normally distributed noise to our estimation of actions
        # whos values are already in [-1,1]

    def forward(self, x):
        return self.mu(x)


class ModelCritic(nn.Module):
    def __init__(self, obs_size):
        super(ModelCritic, self).__init__()

        self.value = nn.Sequential(
            nn.Linear(obs_size, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, 1)
        )

    def forward(self, x):
        return self.value(x)


class AgentA2C(ptan.agent.BaseAgent):
    def __init__(self, net, device="cpu"):
        self.net = net
        self.device = device

    def __call__(self, states, agent_states):
        states_v = ptan.agent.float32_preprocessor(states).to(self.device)

        mu_v = self.net(states_v)
        mu = mu_v.data.cpu().numpy()
        logstd = self.net.logstd.data.cpu().numpy()
        actions = mu + np.exp(logstd) * np.random.normal(size=logstd.shape)  # добавляем нормально распределённый шум
        actions = np.clip(actions, -1, 1)  # обрезаем веса так, чтобы результат не выходил за пределы [-1,1]
        return actions, agent_states

