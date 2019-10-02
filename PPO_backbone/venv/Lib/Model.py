import ptan
import numpy as np
import torch
import torch.nn as nn
import random
import torch.optim as optim

ALPHA = 0.1

HID_SIZE=64

# Предполагаем, что Actor вычисляет advantages
#Critic вычисляет V(s)
#A2C is just an agent, which returns actions, that he would take in particular state
#

class ModelActor_1(nn.Module):
    def __init__(self, obs_len, actions_n, p=0.0, num_envs = 16):
        super(ModelActor_1, self).__init__()

        self.length = 25
        self.forwards = 0
        self.p = p
        self.num_envs = num_envs
        self.envs = False

        self.hidden_dim_11 = obs_len * 3
        self.hidden_dim_12 = obs_len
        self.num_layers_11 = 1  # 2
        self.num_layers_12 = 1

        self.hidden_11 = (torch.zeros((self.num_layers_11, 1, self.hidden_dim_11), device="cuda"),
                          torch.zeros((self.num_layers_11, 1, self.hidden_dim_11), device="cuda"))
        self.logstd = nn.Parameter(torch.zeros(2)) # some sort of parameter in A2c down below with it's help we add normally distributed noise to our estimation of actions
        #whos values are already in [-1,1]

        self.drop = nn.Dropout(p=self.p)
        self.leak = nn.LeakyReLU(negative_slope=0.2)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        # advances
        self.linear_11 = nn.Linear(obs_len, obs_len * 2)  # 1
        self.linear_12 = nn.Linear(obs_len * 3, 64)  # # 3 obs_len->64
        self.linear_13 = nn.Linear(obs_len, 64)  # 5
        self.linear_14 = nn.Linear(64, actions_n)  # 6

        self.lstm_11 = nn.LSTM(input_size=obs_len, hidden_size=self.hidden_dim_11,
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
            """
            self.hidden_12 = (torch.zeros((self.num_layers_12, 1, self.hidden_dim_12), device="cuda"),
                              torch.zeros((self.num_layers_12, 1, self.hidden_dim_12), device="cuda"))
                              """
            if self.envs:
                self.dict[self.name.value] = self.hidden_11

    def set_envs(self, envs, name, change):
            self.envs = True
            self.dict = {env.name: (torch.zeros((self.num_layers_11, 1, self.hidden_dim_11), device="cuda"),
                                     torch.zeros((self.num_layers_11, 1, self.hidden_dim_11), device="cuda")) for env in envs}
            self.name = name
            self.change = change

    def forward(self, var):
        #print(var)
        if True:
            self.forwards += 1
            if (self.forwards == -1):
                self.reset()
            if (len(var.shape) == 2):
                x = var.unsqueeze(1)
            elif(len(var.shape) == 1):
                var = var.unsqueeze(0)
                x = var.unsqueeze(1)
            else:
                x = var
            if self.envs and self.change:
                self.hidden_11 = self.dict[self.name.value]
            #adv = self.linear_11(x)
            #adv = self.relu(adv)
            adv, self.hidden_11 = self.lstm_11(x, self.hidden_11)
            adv = adv.view(-1, self.hidden_dim_11)
            adv = self.relu(adv)
            adv = self.linear_12(adv)
            adv = self.relu(adv)  # leak
            """adv = adv.unsqueeze(1)
            adv, self.hidden_12 = self.lstm_12(adv, self.hidden_12)
            adv = adv.view(-1, self.hidden_dim_12)
            adv = self.leak(adv)"""
            #adv = self.drop(adv)
            """
            adv = self.linear_13(adv)
            adv = self.relu(adv)
            adv = self.drop(adv)"""
            adv = self.linear_14(adv)
            #adv = self.tanh(adv)

            if self.envs:
                self.dict[self.name] = self.hidden_11
            kek = adv.data.cpu().numpy()
            if(np.isnan(kek).any()):
                print("NAN value")
            if torch.isnan(adv).any():
                print("Random torch")
                k = random.randint(0,2)
                mas = np.zeros((1,3))
                mas[0][k] = 1.0
                adv = torch.FloatTensor(mas).to("cuda")
        #print(adv)
        return adv

class ModelActor_2(nn.Module):
    def __init__(self, obs_len, actions_n, p=0.0, num_envs = 16):
        super(ModelActor_2, self).__init__()

        self.length = 15
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
            """
            self.hidden_12 = (torch.zeros((self.num_layers_12, 1, self.hidden_dim_12), device="cuda"),
                              torch.zeros((self.num_layers_12, 1, self.hidden_dim_12), device="cuda"))
                              """
            if self.envs:
                self.dict[self.name.value] = self.hidden_11

    def set_envs(self, envs, name, change):
            self.envs = True
            self.dict = {env.name: (torch.zeros((self.num_layers_11, 1, self.hidden_dim_11), device="cuda"),
                                     torch.zeros((self.num_layers_11, 1, self.hidden_dim_11), device="cuda")) for env in envs}
            self.name = name
            self.change = change

    def forward(self, kek):
        print(kek)
        if len(kek.shape)==3:
            var = kek[0][0][:126].view((1,1,-1))
        else:
            var = kek[0][:126].view((1, 1, -1))
        if True:
            self.forwards += 1
            if (self.forwards == -1):
                self.reset()
            if (len(var.shape) == 2):
                x = var.unsqueeze(1)
            else:
                x = var
            if self.envs and self.change:
                self.hidden_11 = self.dict[self.name.value]
            print(self.hidden_11)
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
            adv = self.tanh(adv)

            if self.envs:
                self.dict[self.name.value] = self.hidden_11

        return adv


class ModelCritic(nn.Module):
    def __init__(self, obs_len, p=0.0, num_envs = 16):
        super(ModelCritic, self).__init__()

        self.length = 15
        self.forwards = 0
        self.p = p
        self.num_envs = num_envs
        self.envs = False

        self.num_layers_21 = 1
        self.hidden_dim_21 = obs_len * 3
        self.hidden_21 = (torch.zeros((self.num_layers_21, 1, self.hidden_dim_21), device="cuda"),
                          torch.zeros((self.num_layers_21, 1, self.hidden_dim_21), device="cuda"))

        self.drop = nn.Dropout(p=self.p)
        self.leak = nn.LeakyReLU(negative_slope=0.2)
        self.relu = nn.ReLU()

        self.linear_21 = nn.Linear(obs_len, obs_len * 2)  # 1

        self.lstm_21 = nn.LSTM(input_size=obs_len, hidden_size=self.hidden_dim_21,
                               num_layers=self.num_layers_21, batch_first=True)

        self.linear_22 = nn.Linear(obs_len * 2, obs_len)  # 3
        self.linear_23 = nn.Linear(3* obs_len, 64)  # 5 #obs_len->*2
        self.linear_24 = nn.Linear(64, 1)

    def rise_length(self):
        self.length += 2

    def vanish_length(self):
        self.length = 10000000000

    def reset(self):
        self.forwards = 0

        self.hidden_21 = (torch.zeros((self.num_layers_21, 1, self.hidden_dim_21), device="cuda"),
                          torch.zeros((self.num_layers_21, 1, self.hidden_dim_21), device="cuda"))
        if self.envs:
            self.dict[self.name] = self.hidden_21

    def set_envs(self, envs, name, change):
        self.envs = True
        self.dict = {env.name: (torch.zeros((self.num_layers_11, 1, self.hidden_dim_21), device="cuda"),
                                torch.zeros((self.num_layers_11, 1, self.hidden_dim_21), device="cuda")) for env in envs}
        self.name = name
        self.change = change
        
        
    def forward(self, var):
        #print(var)
        self.forwards += 1
        if (self.forwards == -1):
            self.reset()
        if (len(var.shape) == 2):
            x = var.unsqueeze(1)
        else:
            x = var
        if self.envs and self.change:
            self.hidden_21 = self.dict[self.name]

        #val = self.linear_21(x)
        #val = self.leak(val)
        val, self.hidden_21 = self.lstm_21(x, self.hidden_21)
        val = val.view(-1, self.hidden_dim_21)
        val = self.relu(val)
        """
        val = self.linear_22(val)
        val = self.relu(val)
        val = self.drop(val)"""
        val = self.linear_23(val)
        val = self.relu(val)
        val = self.drop(val)
        val = self.linear_24(val)

        if self.envs:
            self.dict[self.name] = self.hidden_21
            
        return val

class model_last_layer(nn.Module):
    def __init__(self):
        super(model_last_layer, self).__init__()
        self.linear_1 = nn.Linear(5,10)
        self.linear_2 = nn.Linear(10,3)

    def reset(self):
        pass

    def forward(self, var):
        #print(var)
        out = self.linear_1(var)
        out = self.linear_2(out)
        return out

class AgentA2C(ptan.agent.BaseAgent):
    def __init__(self, net_1, device="cpu"):
        self.net = net_1
        #self.net_2 = net_2
        #self.net_2 = net_2
        self.device = device
        #self.coef = 0.9
        #self.num = 0

    def get_coef(self):
        return self.coef

    def dec_coef(self):
        if(self.num<20000000000):
            self.coef = 0.9-self.num*0.6/(20000000000)

    def __call__(self, states, agent_states):
        states_v = ptan.agent.float32_preprocessor(states).to(self.device)
        """
        mu_v_2 = self.net_2(states_v)
        mu_v = self.net(states_v)
        mu = mu_v.data.cpu().numpy()
        mu_2 = mu_v_2.data.cpu().numpy()
        logstd = self.net.logstd.data.cpu().numpy()
        actions = mu*(1-self.coef) + mu_2*self.coef + np.exp(logstd) * np.random.normal(size=logstd.shape) # добавляем нормально распределённый шум
        self.num+=1
        self.dec_coef()
        """
        #mu_1 = torch.FloatTensor(states[0][0][:-3]).to("cuda")
        #mu_2 = torch.FloatTensor(states[0][0][-3:]).to("cuda")
        #mu_2 = mu_2.view((1,-1))
        mu_v = self.net(torch.FloatTensor(states).to("cuda"))
        #mu_v = torch.cat((mu_v,mu_2), dim=-1)
        #actions= self.net_2(mu_v)
        actions = mu_v.data.cpu().numpy()
        #logstd = self.net.logstd.data.cpu().numpy()
        #actions = actions + np.exp(logstd) * np.random.normal(size=logstd.shape)
        actions = np.clip(actions, -1, 1) # обрезаем веса так, чтобы результат не выходил за пределы [-1,1]
        if (np.isnan(actions[0]).any()):
            act = random.randint(0, 2)
            print("random")
            for i in range(3):
                actions[0][i] = 0.0
                if (i == act):
                    actions[0][i] = 1.0
        #print(actions)
        if(torch.isnan(self.net.hidden_11[0]).any()):
            print("nan")
        return actions, agent_states





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

        self.hidden_12 = (torch.zeros((self.num_layers_12, 1, self.hidden_dim_12), device="cuda"),
                          torch.zeros((self.num_layers_12, 1, self.hidden_dim_12), device="cuda"))

        self.num_layers_21 = 1
        self.hidden_dim_21 = obs_len * 2
        self.hidden_21 = (torch.zeros((self.num_layers_21, 1, self.hidden_dim_21), device="cuda"),
                          torch.zeros((self.num_layers_21, 1, self.hidden_dim_21), device="cuda"))

        self.drop = nn.Dropout(p=self.p)
        self.leak = nn.LeakyReLU(negative_slope=0.2)
        self.relu = nn.ReLU()

        # advances
        self.linear_11 = nn.Linear(obs_len, obs_len * 2)  # 1
        self.linear_12 = nn.Linear(obs_len * 2, 64)  # # 3 obs_len->64
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
        self.linear_23 = nn.Linear(2 * obs_len, 64)  # 5 #obs_len->*2
        self.linear_24 = nn.Linear(64, 1)

    def rise_length(self):
        self.length += 2

    def reset(self):
        self.forwards = 0
        self.hidden_11 = (torch.zeros((self.num_layers_11, 1, self.hidden_dim_11), device="cuda"),
                          torch.zeros((self.num_layers_11, 1, self.hidden_dim_11), device="cuda"))

        self.hidden_12 = (torch.zeros((self.num_layers_12, 1, self.hidden_dim_12), device="cuda"),
                          torch.zeros((self.num_layers_12, 1, self.hidden_dim_12), device="cuda"))

        self.hidden_21 = (torch.zeros((self.num_layers_21, 1, self.hidden_dim_21), device="cuda"),
                          torch.zeros((self.num_layers_21, 1, self.hidden_dim_21), device="cuda"))
        if self.envs:
            self.dict[self.name] = [self.hidden_11, self.hidden_12, self.hidden_21]

    def set_envs(self, envs, name, change):
        self.envs = True
        self.dict = {env.name: [(torch.zeros((self.num_layers_11, 1, self.hidden_dim_11), device="cuda"),
                                 torch.zeros((self.num_layers_11, 1, self.hidden_dim_11), device="cuda")),

                                (torch.zeros((self.num_layers_11, 1, self.hidden_dim_11), device="cuda"),
                                 torch.zeros((self.num_layers_11, 1, self.hidden_dim_11), device="cuda")),

                                (torch.zeros((self.num_layers_11, 1, self.hidden_dim_11), device="cuda"),
                                 torch.zeros((self.num_layers_11, 1, self.hidden_dim_11), device="cuda"))

                                ] for env in envs}
        self.name = name
        self.change = change

    def forward(self, var):
        self.forwards += 1
        if (self.forwards == 100000):
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