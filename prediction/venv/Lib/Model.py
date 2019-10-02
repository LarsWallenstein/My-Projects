import ptan
import numpy as np
import torch
import torch.nn as nn
import random
import torch.optim as optim

ALPHA = 0.1

HID_SIZE=64



class Model_1(nn.Module):
    def __init__(self, obs_len, actions_n, p=0.0, num_envs = 16):
        super(Model_1, self).__init__()

        self.length = 10
        self.forwards = 0
        self.p = p
        self.num_envs = num_envs
        self.envs = False

        self.hidden_dim_11 = obs_len * 2
        self.hidden_dim_12 = obs_len
        self.num_layers_11 = 1  # 2
        self.num_layers_12 = 1

        self.hidden_11 = (torch.zeros((self.num_layers_11, 1, self.hidden_dim_11), device="cuda"),
                          torch.zeros((self.num_layers_11, 1, self.hidden_dim_11), device="cuda"))
        self.logstd = nn.Parameter(torch.zeros(actions_n)) # some sort of parameter in A2c down below with it's help we add normally distributed noise to our estimation of actions
        #whos values are already in [-1,1]

        self.drop = nn.Dropout(p=self.p)
        self.leak = nn.LeakyReLU(negative_slope=0.2)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        # advances
        self.linear_11 = nn.Linear(obs_len, obs_len * 2)  # 1
        self.linear_12 = nn.Linear(obs_len * 2, 64)  # # 3 obs_len->64
        self.linear_13 = nn.Linear(obs_len, 64)  # 5
        self.linear_14 = nn.Linear(64, actions_n)  # 6

        self.lstm_11 = nn.LSTM(input_size=obs_len * 2, hidden_size=self.hidden_dim_11,
                               num_layers=self.num_layers_11, batch_first=True)  # 2
        """
        self.lstm_12 = nn.LSTM(input_size=obs_len, hidden_size=self.hidden_dim_12,
                               num_layers=self.num_layers_12, batch_first=True)  # 4 #"""

    def rise_length(self):
            self.length += 2

    def vanish_length(self):
        self.length = 20

    #ресетится только текущий env
    def reset(self):
            self.forwards = 0
            self.hidden_11 = (torch.zeros((self.num_layers_11, 1, self.hidden_dim_11), device="cuda"),
                              torch.zeros((self.num_layers_11, 1, self.hidden_dim_11), device="cuda"))


    def forward(self, var):
        if True:
            self.forwards += 1
            if (self.forwards == -1):
                self.reset()
            if (len(var.shape) == 2):
                x = var.unsqueeze(1)
            else:
                x = var
            print(x)
            print(x[0][0][:-3])
            adv = self.linear_11(x)
            adv = self.leak(adv)
            adv, self.hidden_11 = self.lstm_11(adv, self.hidden_11)
            adv = adv.view(-1, self.hidden_dim_11)
            adv = self.leak(adv)
            adv = self.linear_12(adv)
            adv = self.relu(adv)  # leak
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
            #adv = self.tanh(adv)


        return adv