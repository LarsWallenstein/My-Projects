import os
import math
import ptan
import gym
from tensorboardX import SummaryWriter

import Model
from Env import StocksEnv

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

ENV_ID = "Roboschoo;HalfCheetah-v1"
GAMMA = 0.99
GAE_LAMBDA = 0.95

TRAJECTORY_SIZE = 2049
LEARNING_RATE_ACTOR = 1e-4
LEARNING_RATE_CRITIC = 1e-3

PPO_EPS = 0.2
PPO_EPOCHES = 10
PPO_BATCH_SIZE = 64

TEST_ITERS = 1000


def test_net(net, env, count=10, device="cuda"):
    rewards = 0.0
    steps = 0
    for _ in range(count):
        obs = env.reset()
        while True:
            obs_v = ptan.agent.float32_preprocessor([obs]).to(device)
            mu_v = net(obs_v)[0]
            action = mu_v.squeeze(dim=0).data.cpu().numpy()
            action = np.clip(action, -1, 1)
            obs, reward, done, _ = env.step(action)
            rewards += reward
            steps += 1
            if done:
                break
    return rewards / count, steps / count


def calc_logprob(mu_v, logstd, actions_v):
    p1 = -((mu_v - actions_v) ** 2) / (2 * torch.exp(logstd_v).clamp(min=1e-3))
    # clamp keeps tensor in between [min,max]
    p2 = -torch.log(torch.sqrt(2 * math.pi * torch.exp(logstd_v)))
    return p1 + p2


# озвращает advantages and Q-values of states вычисленные на основании critic
def calc_adv_ref(trajectory, net_crt, states_v, device="cuda"):
    """
    :param trajectory: trajectory list
    :param net_crt: critic network
    :param states_v: states tensor
    :param device:
    :return: tuple with advantage numpy array and reference values
    """
    values_v = net_crt(states_v)
    values = values_v.squeeze().data.cpu().numpy()
    # converting tensor to numpy array
    # generalised advantage estimator: smoothed version of the advantage
    last_gae = 0.0
    result_adv = []
    result_ref = []

    for val, next_val, (exp,) in zip(reversed(values[:-1]), reversed(values[1:]),
                                     reversed(trajectory[:-1])):  # тут посчитанные критиком значения состояний
        if exp.done:
            # б_t = r_t - V(s_t)
            delta = exp.reward - val
            last_gae = delta
        else:
            # б_t = r_t + j * V(s_[t + 1]) - V(s_t)
            # _
            # A_t = б_t + ( j * a) * б_[t+1] + ... + ( j * a)^(T-t+1) * б_[ T-1 ]
            delta = exp.reward + GAMMA * next_val - val
            last_gae = delta + GAMMA * GAE_LAMBDA * last_gae
        # last_gae - advantage estimation
        # result_ref - A + V = Q
        result_adv.append(last_gae)
        result_ref.append(last_gae + val)

    adv_v = torch.FloatTensor(list(reversed(result_adv))).to(device)
    ref_v = torch.FloatTensor(list(reversed(result_ref))).to(device)
    return adv_v, ref_v


def make_env(net, i, flag):
    stk = StocksEnv(net=net)
    stk.set_name(str(i))
    if (flag):
        stk._state.set_length(net.length)
    return stk


def exp_b(net, name, change, device, flag=False):
    envs = [make_env(net, p, flag) for p in range(net.num_envs)]
    net.set_envs(envs, name, change)
    # agent = ptan.agent.PolicyAgent(lambda x: net(x)[0], apply_softmax=True, device=device)
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
    i = 1
    iter = 0
    rew = -200
    idx = -1
    op = False
    max_obtained_reward = 0.2
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
    ########################################################################################################################################################################################################################

    device = "cuda"
    save_path = os.path.join("saves", "ppo-" + "name")
    os.makedirs(save_path, exist_ok=True)

    env = gym.make(ENV_ID)
    test_env = gym.make(ENV_ID)

    net_act = Model.ModelActor(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    net_crt = Model.ModelCritic(env.observation_space[0]).to(device)
    print(net_act, net_crt)

    writer = SummaryWriter(comment="-ppo_" + "name")
    # agent = Model.AgentA2C(net_act, device=device)
    # exp_source = ptan.experience.ExperienceSource(env, agent, steps_count=1)# обрабатываем по одному опыту за раз (state, action, reward, done)

    opt_act = optim.Adam(net_act.parameters(), lr=LEARNING_RATE_ACTOR)
    opt_crt = optim.Adam(net_crt.parameters(), lr=LEARNING_RATE_CRITIC)

    set_traj = []
    trajectory = []
    best_reward = None

    exp_buf = exp_b(net, name, change, device)
    exp_source = exp_buf[j.value]

    with ptan.common.utils.RewardTracker(writer) as tracker: