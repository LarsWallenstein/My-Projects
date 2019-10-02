import gym
import os
import torch
import csv
import glob
import numpy as np
import collections
import enum
import random
import pandas as pd
import sys
import time
import ptan


class Actions(enum.Enum):
    Skip = 0
    Buy = 1
    B_Buy = 2
    Sell = 3
    S_Sell = 4


def csv_to_txt(string):
    subStrOld = "csv"
    subStrNew = "txt"
    lenStrOld = len(subStrOld)

    while string.find(subStrOld) > 0:
        i = string.find(subStrOld)
        string = string[:i] + subStrNew + string[i + lenStrOld:]
    return string

def load_data():  # 1. data, 2.prices
    time_frame = str(random.randint(1,2))
    der = str(2)  # str(random.randint(1,2))

    frame = {
        '1': "1 Day",
        '2': "1 Hour",
        '3': "5 Min"
    }

    eq = {
        '1': "Stocks",
        '2': "ETFs"
    }

    time_frame = frame[time_frame]
    der = eq[der]
    """
    frame = {
        # '1':"1 Day",
        '1': "1 Hour",
        '2': "5 Min"
    }
    eq = {
        '1': "Stocks",
        '2': "ETFs"
    }
    time_frame = frame[str(random.randint(1, 2))]
    if time_frame == '1 Hour':
        der = 'Stocks'
    else:
        der = eq[str(random.randint(1, 2))]
        """
    PATH = "C:/Users/corsa/Documents/Stocks_2_new/"+time_frame+"/"+der
    PATH_2 = "C:/Users/corsa/Documents/Stocks/"+time_frame+"/"+der
    #PATH = 'C:/Users/corsa/Documents/Stocks_2/' + time_frame + '/' + der
    #PATH_2 = 'C:/Users/corsa/Documents/Stocks/' + time_frame + '/' + der
    # PATH = '/content/my_drive/My Drive/Stocks_2/' + time_frame + '/' + der
    # PATH_2 = '/content/my_drive/My Drive/Stocks/' + time_frame + '/' + der
    stocks = os.listdir(path=PATH)
    while True:
        num = random.randint(0, len(stocks) - 1)
        #d1 = np.array(pd.read_csv(PATH + '/' + stocks[num], sep=',').round(5))
        d1 = np.array(pd.read_csv(PATH+ '/' + stocks[num], sep=',').round(5))
        if not (np.isnan(d1).any() or np.isinf(d1).any()):
            break
    # return np.array(pd.read_csv(PATH,sep=',').round(5)), pd.read_csv(PATH_2,sep=',').round(5)
    #print(stocks[num], time_frame, der)

    return d1, pd.read_csv(PATH_2+ '/' + csv_to_txt(stocks[num]), sep=',').round(5)# + '/' + csv_to_txt("tvix.us.csv"), sep=',').round(5)

class RewardTracker:
    def __init__(self, writer, stop_reward, num_of_envs=64):
        self.writer = writer
        self.stop_reward = stop_reward
        self.hist = -11
        self.save = False
        self.last_mean = -11
        self.num_of_envs = num_of_envs

    def __enter__(self):
        self.ts = time.time()
        self.ts_frame = 0
        self.total_rewards = []
        return self

    def __exit__(self, *args):
        # self.writer.close()
        pass

    def reward(self, reward, frame, epsilon=None):
        self.save = False
        self.total_rewards.append(reward)
        speed = (frame - self.ts_frame) / (time.time() - self.ts)
        self.ts_frame = frame
        self.ts = time.time()
        mean_reward = np.mean(self.total_rewards[-self.num_of_envs:])
        self.last_mean = mean_reward
        if mean_reward > self.hist:
            self.hist = mean_reward
            self.save = True
        """if len(self.hist)<5:
            self.hist.append(mean_reward)
        if len(self.hist)==5 and not self.save:
            self.hist = np.array(self.hist)
        if frame>5 and len(self.hist)==5:
            self.save = False
            if mean_reward>np.mean(self.hist):
                self.hist[0:-1] = self.hist[1:]
                self.hist[-1] = mean_reward
                if mean_reward>np.mean(self.hist[-3:]):

        """
        epsilon_str = "" if epsilon is None else ", eps %.2f" % epsilon
        print("%d: done %d games, mean reward %.3f, speed %.2f f/s%s" % (
            frame, len(self.total_rewards), mean_reward, speed, epsilon_str
        ))
        sys.stdout.flush()

        if epsilon is not None:
            self.writer.add_scalar("epsilon", epsilon, frame)
        self.writer.add_scalar("speed", speed, frame)
        self.writer.add_scalar("reward_100", mean_reward, frame)
        self.writer.add_scalar("reward", reward, frame)
        if mean_reward > self.stop_reward:
            print("Solved in %d frames!" % frame)
            return True
        return mean_reward
        """
        if epsilon is not None:
            self.writer.save_value("Graph", "epsilon", frame, epsilon)
        self.writer.save_value("Graph", "speed", frame, speed)
        self.writer.save_value("Graph", "reward_100", frame, mean_reward)
        self.writer.save_value("Graph", "reward", frame, reward)
        if mean_reward > self.stop_reward:
            print("Solved in %d frames!" % frame)
            return True
            """

def unpack_batch(batch, net, last_val_gamma, device='cuda'):
    """
    Convert batch into training tensors
    :param batch:
    :param net:
    :return: states variable, actions tensor, reference values variable
    """
    states = []
    actions = []
    rewards = []
    not_done_idx = []
    last_states = []
    for idx, exp in enumerate(batch):
        states.append(np.array(exp.state, copy=False))
        actions.append(int(exp.action))
        rewards.append(exp.reward)
        if exp.last_state is not None:
            not_done_idx.append(idx)
            last_states.append(np.array(exp.last_state, copy=False))
    states_v = torch.FloatTensor(states).to(device)
    actions_t = torch.LongTensor(actions).to(device)

    # handle rewards
    rewards_np = np.array(rewards, dtype=np.float32)
    if not_done_idx:
        last_states_v = torch.FloatTensor(last_states).to(device)
        last_vals_v = net(last_states_v)[1]
        last_vals_np = last_vals_v.data.cpu().numpy()[:, 0]
        rewards_np[not_done_idx] += last_val_gamma * last_vals_np

    ref_vals_v = torch.FloatTensor(rewards_np).to(device)
    return states_v, actions_t, ref_vals_v