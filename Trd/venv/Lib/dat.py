import ptan
import time
from Env import StocksEnv
import torch.multiprocessing as mp
import collections
import numpy as np
import random
from ctypes import c_bool, c_char_p

NUM_ENVS = 16
GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01
BATCH_SIZE = 10

REWARD_STEPS = 1
CLIP_GRAD = 0.1

PROCESSES_COUNT = 1

def make_env(net, i):
    stk = StocksEnv(net=net)
    stk.set_name(str(i))
    return stk

def data_func(net, device, train_queue, i, name, change, j):
    TotalReward = collections.namedtuple('TotalReward', field_names='reward')
    envs = [make_env(net,p) for p in range(net.num_envs)]
    net.set_envs(envs, name, change)
    lock = mp.Lock()
    agent = ptan.agent.PolicyAgent(lambda x: net(x)[0], apply_softmax=True, device=device)
    exp_buf = []
    for l in range(NUM_ENVS):
        exp_source = ptan.experience.ExperienceSourceFirstLast(envs[l], agent, gamma=GAMMA, steps_count=REWARD_STEPS)
        exp_buf.append(exp_source)
    while True:
        with lock:
            if i.value==-1:
                break
        exp_source = exp_buf[j.value]
        for exp in exp_source:
            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                with lock:
                    j.value=(j.value+1)%NUM_ENVS
                    name.value = str(j.value)
                    change.value = True
                train_queue.put(TotalReward(reward=np.mean(new_rewards)))
            train_queue.put(exp)
            if train_queue.full():
                with lock:
                    i.value = 2
                    print("full")
            while True:
              with lock:
                tmp = not (i.value==2)
              if(tmp):
                break
              else:
                time.sleep(0.2)
            with lock:
                if i.value==-1:
                    break


def data_func_old(net, device, train_queue, i, name, change, j):
    TotalReward = collections.namedtuple('TotalReward', field_names='reward')
    envs = [make_env(net,p) for p in range(net.num_envs)]
    net.set_envs(envs, name, change)
    lock = mp.Lock()
    agent = ptan.agent.PolicyAgent(lambda x: net(x)[0], apply_softmax=True, device=device)
    exp_buf = []
    for l in range(NUM_ENVS):
        exp_source = ptan.experience.ExperienceSourceFirstLast(envs[l], agent, gamma=GAMMA, steps_count=REWARD_STEPS)
        exp_buf.append(exp_source)
    while True:
        with lock:
            if i.value==-1:
                break
        exp_source = exp_buf[j.value]
        for exp in exp_source:
            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                with lock:
                    j.value=(j.value+1)%NUM_ENVS
                    name.value = str(j.value)
                    change.value = True
                train_queue.put(TotalReward(reward=np.mean(new_rewards)))
            train_queue.put(exp)
            if train_queue.full():
                with lock:
                    i.value = 2
            with lock:
                if i.value==-1:
                    break




