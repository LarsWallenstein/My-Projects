import dat
from Env import StocksEnv
import collections
import ptan
import torch.multiprocessing as mp
import Model as m
import random
import time
from Model import DDQN
from ctypes import c_bool, c_char_p
import numpy as np

train_queue = [None]
reward_queue = [None]
batch = []
step_idx = 0
lol = 0
i = 1
idx=-1
op = False
name = mp.Manager().Value(c_char_p, "0")  # contsains the name of thee current environment
change = mp.Manager().Value(c_bool, False)  # contains the value for changing to new env
j = mp.Value('i', 1)
    # to control current state of the process
    # 2-reading from queue,
    # -1 - end of the process
    # 1-writing to process

def make_env(net, i):
    stk = StocksEnv(net=net)
    stk.set_name(str(i))
    return stk
device ="cuda"
GAMMA = 0.99
LEARNING_RATE = 0.005
ENTROPY_BETA = 0.01
BATCH_SIZE = 1

REWARD_STEPS = 3
CLIP_GRAD = 0.25

PROCESSES_COUNT = 1
NUM_ENVS = 16

REWARD_BOUND = 1000
CHECKPOINT_EVERY_STEP = 1000

net = DDQN(126, 3).to(device)
TotalReward = collections.namedtuple('TotalReward', field_names='reward')
if __name__=="__main__":
    envs = [make_env(net, p) for p in range(net.num_envs)]
    net.set_envs(envs, name, change)
    agent = ptan.agent.PolicyAgent(lambda x: net(x)[0], apply_softmax=True, device=device)
    exp_buf = []
    lock=mp.Lock()
    for l in range(NUM_ENVS):
        exp_source = ptan.experience.ExperienceSourceFirstLast(envs[l], agent, gamma=GAMMA,steps_count=REWARD_STEPS)
        exp_buf.append(exp_source)
    while True:
        if i == -1:
            break
        exp_source = exp_buf[j.value]
        flag=False
        for exp in exp_source:
            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                with lock:
                    j.value = (j.value + 1) % NUM_ENVS
                    name.value = str(j.value)
                    change.value = True
                print("rewards + experience:")
                print(new_rewards,exp)
                reward_queue[0]=TotalReward(reward=np.mean(new_rewards))
                break
            print("Experience:")
            print(exp)
        #train_queue[0]=exp
"""

if __name__ == '__main__':
    device = "cuda"
    NUM_ENVS = 8
    mp.set_start_method('spawn', force=True)
    net = m.DDQN(126, 3, 0.1, num_envs=NUM_ENVS).to(device)
    net.share_memory()
    i = mp.Value('i', 1)  # to control current state of the process
                        # 2-reading from queue,
                        # -1 - end of the process
                        # 1-writing to process
    name = mp.Manager().Value(c_char_p, "0")  # contsains the name of thee current environment
    change = mp.Manager().Value(c_bool, False)  # contains the value for changing to new env
    j = mp.Value('i', 0)
    lock = mp.Lock()
    train_queue = mp.Queue(maxsize=3)
    data_proc = mp.Process(target=dat.data_func, args=(net, device, train_queue, i, name, change, j))
    data_proc.start()
    ii = 0
    while True:
        if ii == 3:
            with lock:
                i.value = -1
        time.sleep(0.005)
        with lock:
            if (i.value == 2):
                train_entry = train_queue.get()
                print(train_entry)  # for testing
                ii += 1
            if (i.value == -1):
                break
    print("p")
    data_proc.join()
    """
