from Common import *
from Model import *
from Env import *
import torch.nn.utils as nn_utils
import torch.nn.functional as F
import tensorboardX as tb
from ctypes import c_bool, c_char_p

GAMMA = 0.98
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01
BATCH_SIZE = 1

REWARD_STEPS = 1
CLIP_GRAD = 0.2

PROCESSES_COUNT = 1
NUM_ENVS = 2

REWARD_BOUND = 1000
CHECKPOINT_EVERY_STEP = 1000

import ptan
import torch.multiprocessing as mp

def make_env(net, i):
    stk = StocksEnv(net=net)
    stk.set_name(str(i))
    return stk

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    device = "cuda"
    PATH = 'C:/Users/corsa/Documents/Stocks_2/'
    NAME = "TRADE"
    name = "1"
    #writer = tb.SummaryWriter(comment="-a3c-data_" + NAME + "_" + name)
    #saves_path = PATH + '/' + str(len(os.listdir(path=PATH)) - 3)
    #os.makedirs(saves_path, exist_ok=True)


    net = DDQN(126, 3, num_envs=NUM_ENVS).to(device)
    net.share_memory()
    """
    load_path = "C:/Users/corsa/Documents/ppo_checkpoints"
    net.load_state_dict(torch.load(os.path.join(load_path, "checkpoint-4.82-%3d.data" % 97)))
    net.reset()"""
    load_path = "C:/Users/corsa/Documents/Stocks_2/59/"
    net.load_state_dict(torch.load(os.path.join(load_path, "checkpoint-%3d.data" % 144)))
    # net.load_state_dict(torch.load(os.path.join(load_path, "checkpoint1.606726802367143-138")))
    net.reset()
    #optimizer = optim.Adadelta(net.parameters(), lr=0.1) # Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3)

    train_queue = [None]
    reward_queue = [None]
    batch = []
    step_idx = 0
    lol = 0
    i = 100
    iter = 0
    idx=-1
    op = False
    writer = tb.SummaryWriter(comment="-a3c-data_" + NAME + "_" + name)
    # to control current state of the process
    # 2-reading from queue,
    # -1 - end of the process
    # 1-writing to process
    name = mp.Manager().Value(c_char_p, "0")  # contsains the name of thee current environment
    change = mp.Manager().Value(c_bool, False)  # contains the value for changing to new env
    j = mp.Value('i', 1)
    lock = mp.Lock()
    top = False
    TotalReward = collections.namedtuple('TotalReward', field_names='reward')
    env=make_env(net,i)
    with RewardTracker(writer, stop_reward=REWARD_BOUND, num_of_envs=NUM_ENVS) as tracker:
        with ptan.common.utils.TBMeanTracker(writer, batch_size=100) as tb_tracker:
            envs = [make_env(net, p) for p in range(net.num_envs)]
            net.set_envs(envs, name, change)
            agent = ptan.agent.PolicyAgent(lambda x: net(x)[0], apply_softmax=True, device=device)
            exp_buf = []
            for l in range(NUM_ENVS):
                exp_source = ptan.experience.ExperienceSourceFirstLast(envs[l], agent, GAMMA,
                                                                       steps_count=REWARD_STEPS)
                exp_buf.append(exp_source)
            while True:
                if i == -1:
                    break
                exp_source = exp_buf[j.value]
                flag = False
                for exp in exp_source:
                    new_rewards = exp_source.pop_total_rewards()
                    if new_rewards:
                        with lock:
                            j.value = (j.value + 1) % NUM_ENVS
                            name.value = str(j.value)
                            change.value = True
                        reward_queue[0] = TotalReward(reward=np.mean(new_rewards))

                    train_queue[0] = exp
                    train_entry = train_queue[0]
                    if reward_queue[0] is not None:
                        i-=1
                        if tracker.reward(reward_queue[0].reward, step_idx):
                            i -= 1
                            break
                        reward_queue[0] = None
                        top = True
                        iter = iter + 1
                    step_idx += 1