import os
import math
import ptan
import gym
#import roboschool
#import argparse
from tensorboardX import SummaryWriter
import torch.multiprocessing as mp
from ctypes import c_bool, c_char_p
import time

from Env import StocksEnv

import Model

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from Common import RewardTracker

GAMMA=0.99
GAE_LAMBDA=0.95

TRAJECTORY_SIZE = 2049
LEARNING_RATE_ACTOR = 1e-4
LEARNING_RATE_CRITIC = 1e-3

PPO_EPS = 0.2
PPO_EPOCHES = 10
PPO_BATCH_SIZE = 64

TEST_ITERS = 1000

NUM_ENVS = 16


def make_env(net, i, flag):
    stk = StocksEnv(net=net)
    stk.set_name(str(i))
    if(flag):
        stk._state.set_length(net.length)
    return stk

def exp_b(net, name, change, device,flag=False):
    envs = [make_env(net, p, flag) for p in range(net.num_envs)]
    net.set_envs(envs, name, change)
    #agent = ptan.agent.PolicyAgent(lambda x: net(x)[0], apply_softmax=True, device=device)
    agent = Model.AgentA2C(net_act, device=device)
    exp_buf = []
    for l in range(NUM_ENVS):
        exp_source = ptan.experience.ExperienceSource(envs[l], agent, steps_count=1)
        exp_buf.append(exp_source)
    return exp_buf


if __name__ == "__main__":
    ####################################################################_____________PARAMETERS________________###########################################################################################################
    train_queue = [None]
    reward_queue = [None]
    batch = []
    step_idx = 0
    lol = 0
    iter = 0
    idx = -1
    op = False
    # to control current state of the process
    # 2-reading from queue,
    # -1 - end of the process
    # 1-writing to process
    name = mp.Manager().Value(c_char_p, "0")  # contsains the name of thee current environment
    change = mp.Manager().Value(c_bool, False)  # contains the value for changing to new env
    j = mp.Value('i', 0)
    lock = mp.Lock()
    top = False
    # TotalReward = collections.namedtuple('TotalReward', field_names='reward')
    ########################################################################################################################################################################################################################

    device = "cuda"


    net_act = Model.ModelActor(150, 3, p=0.0, num_envs=NUM_ENVS).to(device)
    net_crt = Model.ModelCritic(150, num_envs=NUM_ENVS).to(device)

    load_path = "C:/Users/corsa/Documents/ppo_checkpoints/"
    net_act.load_state_dict(torch.load(os.path.join(load_path, "checkpoint_actor-3.9-%3d.data" % 79)))
    net_crt.load_state_dict(torch.load(os.path.join(load_path, "checkpoint_critic-3.9-%3d.data" % 79)))
    net_act.reset()
    net_crt.reset()

    net_act.vanish_length()

    writer = SummaryWriter(comment="-ppo_" + "name")
    exp_buf = exp_b(net_act, name, change, device, flag=True)

    with RewardTracker(writer, 1000, 16) as tracker:
        while True:
            exp_source = exp_buf[j.value]
            for step_idx, exp in enumerate(exp_source):
                rewards_steps = exp_source.pop_rewards_steps()  # вроде как только по окончанию эпизода будут не пустые
                if rewards_steps:
                    rewards, steps = zip(*rewards_steps)
                    kuk = 1.0
                    for r in rewards:
                        kuk*=(1+(r/100.0))
                    print(kuk)
                    time.sleep(5)
                    writer.add_scalar("episode_steps", np.mean(steps), step_idx)
                    mean_reward = tracker.reward(np.mean(rewards), step_idx)
                    iter += 1
                if (exp[0].done):
                    j.value = (j.value + 1) % NUM_ENVS
                    change_trajectory = True
                    exp_source = exp_buf[j.value]