import os
import math
import ptan
import random
from tensorboardX import SummaryWriter
import torch.multiprocessing as mp
from ctypes import c_bool, c_char_p
import time

from Env import StocksEnv

import Model

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


GAMMA=0.99
GAE_LAMBDA=0.95

ENTROPY_BETA = 0.001

TRAJECTORY_SIZE = 2049
LEARNING_RATE_ACTOR = 3e-5
LEARNING_RATE_CRITIC = 1e-4

PPO_EPS = 0.2
PPO_EPOCHES = 10
PPO_BATCH_SIZE = 64

TEST_ITERS = 1000
back = 1

NUM_ENVS = 128

def test_net(net, env, count=10, device="cuda"):
    rewards=0.0
    steps=0
    for _ in range(count):
        obs=env.reset()
        while True:
            obs_v = ptan.agent.float32_preprocessor([obs]).to(device)
            mu_v = net(obs_v)[0]
            action = mu_v.squeeze(dim=0).data.cpu().numpy()
            action = np.clip(action, -1, 1)
            obs, reward, done, _ = env.step(action)
            rewards+=reward
            steps+=1
            if done:
                break
    return rewards/count, steps/count

def calc_logprob(mu_v, logstd_v, actions_v):
    #print(mu_v, actions_v)
    logstd = nn.Parameter(torch.zeros(5)).to('cuda')
    p1 = -((mu_v-actions_v)**2)/(2*torch.exp(logstd).clamp(min=1e-3))
    p2 = -torch.log(torch.sqrt(2*math.pi*torch.exp(logstd)))
    return p1+p2

# озвращает advantages and Q-values of states вычисленные на основании critic
def calc_adv_ref(trajectory, net_crt, states_v, device="cuda"):
    """
    :param trajectory: trajectory list
    :param net_crt: critic network
    :param states_v: states tensor
    :param device:
    :return: tuple with advantage numpy array and reference values
    """
    values_v=torch.FloatTensor().to(device)
    for state in states_v:
        value_v = net_crt(state)
        values_v = torch.cat([values_v,value_v],0)
    #values_v = torch.from_numpy(values_v).float().to(device)
    values = values_v.squeeze().data.cpu().numpy()
    #generalised advantage estimator: smoothed version of the advantage
    last_gae = 0.0
    result_adv=[]
    result_ref = []
    if values.shape != ():
        #print(values)
        for val, next_val, (exp,) in zip(reversed(values[:-1]), reversed(values[1:]), reversed(trajectory[:-1])): # тут посчитанные критиком значения состояний
            if exp.done:
                # б_t = r_t - V(s_t)
                delta = exp.reward - val
                last_gae = delta
            else:
                # б_t = r_t + j * V(s_[t + 1]) - V(s_t)
                # _
                # A_t = б_t + ( j * a) * б_[t+1] + ... + ( j * a)^(T-t+1) * б_[ T-1 ]
                delta = exp.reward+GAMMA*next_val-val
                last_gae = delta+GAMMA*GAE_LAMBDA*last_gae
            #last_gae - advantage estimation
            #result_ref - A + V = Q
            result_adv.append(last_gae)
            result_ref.append(last_gae+val)
    else:
        delta = trajectory[0][0].reward - values
        last_gae = delta
        result_adv.append(last_gae)
        result_ref.append(last_gae + values)

    adv_v = torch.FloatTensor(list(reversed(result_adv))).to(device)
    ref_v = torch.FloatTensor(list(reversed(result_ref))).to(device)
    return adv_v, ref_v

def make_env(net, i, flag):
    stk = StocksEnv(net=net)
    stk.set_name(str(i))
    if(flag):
        stk._state.set_length(net.length)
    return stk

def exp_b(net_1, name, change, device, flag=False):
    envs = [make_env(net_1, p, flag) for p in range(net_1.num_envs)]
    net_1.set_envs(envs, name, change)
    agent = Model.AgentA2C(net_1,device=device)
    exp_buf = []
    for l in range(NUM_ENVS):
        exp_source = ptan.experience.ExperienceSource(envs[l], agent, steps_count=1)
        exp_buf.append(exp_source)
    return exp_buf, agent


if __name__=="__main__":

    ####################################################################_____________PARAMETERS________________###########################################################################################################
    train_queue = [None]
    reward_queue = [None]
    batch = []
    step_idx = 0
    lol = 0
    iter = 0
    idx = -1
    op = False
    name = mp.Manager().Value(c_char_p, "0")  # contsains the name of thee current environment
    change = mp.Manager().Value(c_bool, False)  # contains the value for changing to new env
    j = mp.Value('i', 1)
    lock = mp.Lock()
    top = False
    #TotalReward = collections.namedtuple('TotalReward', field_names='reward')
    ########################################################################################################################################################################################################################
    
    
    device = "cuda"
    save_path = os.path.join("saves","ppo-"+"name")
    os.makedirs(save_path, exist_ok = True)


    net_act_1 = Model.ModelActor_1(150, 5, p=0.0,num_envs=NUM_ENVS).to(device)
    #net_act_2 = Model.model_last_layer().to(device)
    #net_act_2 = Model.ModelActor_2(126, 3, p=0.0,num_envs=NUM_ENVS).to(device)
    net_crt = Model.ModelCritic(150, num_envs=NUM_ENVS).to(device)
    #print(net_act, net_crt)

    writer = SummaryWriter(comment="-ppo_"+"name")
    #agent = Model.AgentA2C(net_act, device=device)
    #exp_source = ptan.experience.ExperienceSource(env, agent, steps_count=1)
    #opt_act_2 = optim.Adam(net_act_2.parameters(), lr=LEARNING_RATE_ACTOR)
    opt_crt = optim.Adam(net_crt.parameters(), lr = LEARNING_RATE_CRITIC)
    opt_act_1 = optim.Adam(net_act_1.parameters(), lr=LEARNING_RATE_ACTOR)
    #load_path = "C:/Users/corsa/PycharmProjects/prediction/checkpoints"
    #net_act_1.load_state_dict(torch.load(os.path.join(load_path, 'best_+0.163_49.dat')))#"checkpoint_actor-4.82-%3d.data" % 97
    # net.load_state_dict(torch.load(os.path.join(load_path, "checkpoint1.606726802367143-138")))
    #load_path = "C:/Users/corsa/PycharmProjects/prediction/checkpoints_2"
    #net_act_2.load_state_dict(torch.load(os.path.join(load_path, 'best_+0.388.dat')))
    net_act_1.reset()
    #net_act_2.reset()

    set_traj = []
    trajectory=[]

    expert_step = []
    estep_traj = []
    set_expert = []

    best_reward = None

    mean_reward = -0.5
    control_point = 0.0



    change_trajectory = False

    exp_buf, agent = exp_b(net_act_1, name, change, device)

    #print("here")
    with ptan.common.utils.RewardTracker(writer) as tracker:
        while True:
            exp_source = exp_buf[j.value]
            for step_idx, exp in enumerate(exp_source): #бежим по experience source
                flag_1 = False
                mean_3  = exp[1][0]
                mean_6 = exp[1][1]
                exp = exp[0]
                long = exp[0][0][0][-3]
                short = exp[0][0][0][-2]


                rewards_steps = exp_source.pop_rewards_steps() # вроде как только по окончанию эпизода будут не пустые
                if rewards_steps:
                    rewards, steps = zip(*rewards_steps)
                    kk = 1.0
                    for reward in rewards:
                        kk*=(1+reward/100.0)
                    print(kk)
                    writer.add_scalar("episode_steps", np.mean(steps), step_idx)
                    mean_reward = tracker.reward(np.mean(rewards), step_idx)
                    iter+=1
                if (exp[0].done):
                    j.value = (j.value + 1) % NUM_ENVS
                    change_trajectory = True
                    exp_source = exp_buf[j.value]
                trajectory.append(exp)# добавляем exp в траекторию

                #EXPERT  ################################
                expert_step = np.array([0,0,0,0,0])#).to(device)
                if long:
                    if(mean_3>0 and mean_6>0):
                        expert_step = np.array([0.6,0.4,0,0,0])#).to(device)#skip
                    elif(mean_3>0 and mean_6<0):
                        expert_step = np.array([0.6, 0.2, 0, 0.2, 0])#).to(device)#skip
                    elif(mean_3<0 and mean_6>0):
                        if(abs(mean_3)>mean_6):
                            expert_step = np.array([0, 0, 0, 1, 0])#).to(device)#sell
                        else:
                            expert_step  = np.array([0.7, 0, 0, 0.3, 0])#).to(device)  # sell
                    else:
                        expert_step = np.array([0,0,0,0,1])#).to(device)#s_sell
                elif short:
                    if (mean_3 > 0 and mean_6 > 0):
                        expert_step = np.array([0, 0, 1, 0, 0])#).to(device)  # b_buy
                    elif (mean_3 > 0 and mean_6 < 0):
                        if(mean_3>abs(mean_6)):
                            expert_step = np.array([0.25, 0.75, 0, 0, 0])#).to(device)  # buy
                        else:
                            expert_step = np.array([0.8, 0.2, 0, 0, 0])#).to(device) # skip
                    elif (mean_3 < 0 and mean_6 > 0):
                        expert_step = np.array([0.8, 0.2, 0, 0, 0])#).to(device)  # sell
                    else:
                        expert_step = np.array([1, 0, 0, 0, 0])#).to(device)  # s_sell
                else:
                    if(mean_3>0 and mean_6>0):
                        expert_step = np.array([0, 1, 0, 0, 0])#).to(device)
                    elif(mean_3<0 and mean_6<0):
                        expert_step = np.array([0, 0, 0, 1, 0])#).to(device)
                    else:
                        expert_step = np.array([0.6, 0.2, 0, 0.2, 0])#).to(device)
                estep_traj.append(expert_step)
                #########################################

                if not change_trajectory:
                    continue
                if change_trajectory:
                    if(len(trajectory)==len(estep_traj) and len(trajectory)>1):
                        set_traj.append(trajectory)
                        set_expert.append(estep_traj)
                    change_trajectory=False
                    trajectory= []
                    estep_traj=[]
                    if len(set_traj)!=NUM_ENVS:
                        break
                if len(set_traj)==NUM_ENVS:
                    flag_1 = True
                    set_traj_states_v = []
                    set_traj_actions_v = []
                    set_traj_adv_v = []
                    set_traj_ref_v = []
                    set_oldlog_prob = []

                    for i in range(len(set_traj)):
                        trajectory = set_traj[i]
                        net_act_1.reset()
                        net_crt.reset()
                        traj_states = [t[0].state for t in trajectory] # t -- [Experience]
                        traj_actions = [t[0].action for t in trajectory]
                        traj_states_v = torch.FloatTensor(traj_states).to(device) # переводим в тензоры
                        traj_actions_v = torch.FloatTensor(traj_actions).to(device)
                        traj_adv_v, traj_ref_v = calc_adv_ref(trajectory, net_crt, traj_states_v, traj_actions_v)
                        mu_v=torch.FloatTensor().to(device)
                        for state_v in traj_states_v:
                            #mu_1 = state_v[0][:-3]
                            #mu_2 = state_v[0][-3:]
                            #mu_2 = mu_2.view((1,-1))
                            actions = net_act_1(state_v)
                            #mu = torch.cat((mu, mu_2), dim=-1)
                            #actions = net_act_2(mu)
                            mu_v = torch.cat([mu_v,actions], 0)
                        old_logprob_v = calc_logprob(mu_v, net_act_1.logstd, traj_actions_v) # как он пишет pi_o_old

                        traj_adv_v = (traj_adv_v - torch.mean(traj_adv_v))/torch.std(traj_adv_v)

                        trajectory = trajectory[:-1]
                        old_logprob_v = old_logprob_v[:-1].detach()

                        set_traj_actions_v.append(traj_actions_v)
                        set_traj_states_v.append(traj_states_v)
                        set_traj_adv_v.append(traj_adv_v)
                        set_traj_ref_v.append(traj_ref_v)
                        set_oldlog_prob.append(old_logprob_v)

                    sum_loss_value = 0.0
                    sum_loss_policy = 0.0
                    count_steps = 0

                    set_loss_value_v = []
                    set_loss_policy_v = []

                    for epoch in range(PPO_EPOCHES):
                        opt_act_1.zero_grad()
                        opt_crt.zero_grad()
                        for i in range(NUM_ENVS):
                            j.value = (j.value + 1) % NUM_ENVS
                            net_act_1.reset()

                            net_crt.reset()

                            states_v = set_traj_states_v[i]
                            actions_v = set_traj_actions_v[i]
                            expert_actions_v = torch.FloatTensor(set_expert[i]).to(device)
                            batch_adv_v = set_traj_adv_v[i].unsqueeze(-1)
                            batch_ref_v = set_traj_ref_v[i]
                            batch_old_logprob_v = set_oldlog_prob[i]


                            values = torch.FloatTensor().to(device)
                            for state_v in states_v:
                                values = torch.cat([values,net_crt(state_v)],0)
                            loss_value_v = F.mse_loss(values.squeeze(-1)[:-1], batch_ref_v)
                            loss_value_v.backward()
                            mu_v = torch.FloatTensor().to(device)
                            for state_v in states_v:
                                actions = net_act_1(state_v)
                                mu_v = torch.cat([mu_v, actions],0)
                            logprob_pi_v = calc_logprob(mu_v, net_act_1.logstd, actions_v)
                            ratio_v = torch.exp(logprob_pi_v[:-1]-batch_old_logprob_v)
                            surr_obj_v = batch_adv_v * ratio_v
                            clipped_surr_v = batch_adv_v*torch.clamp(ratio_v, 1.0-PPO_EPS, 1.0+PPO_EPS)


                            log_prob_v = F.log_softmax(mu_v, dim=1)
                            entropy_loss_v = ENTROPY_BETA * (expert_actions_v * log_prob_v).sum(dim=1).mean()

                            if(iter<150):
                                loss_policy_v = - entropy_loss_v
                            else:
                                if(iter%3==0):
                                    loss_policy_v = -torch.min(surr_obj_v, clipped_surr_v).mean() - entropy_loss_v# - torch.sum(actions_v[:,1:3])/(actions_v[:,1:3].shape[0]*2.0*100.0)
                                else:
                                    loss_policy_v = -torch.min(surr_obj_v, clipped_surr_v).mean()
                            if not (torch.isnan(loss_policy_v)):
                                loss_policy_v.backward()

                            sum_loss_value+=loss_value_v.item()
                            sum_loss_policy+=loss_policy_v.item()
                            count_steps+=1
                            if i ==(NUM_ENVS-1):

                                opt_crt.step()
                                opt_act_1.step()
                                #print("STEP")
                                # policy_loss_v.backward()

                                # opt_act_2.step()
                                # print("step", j.value)

                        #value_loss_v = sum(set_loss_value_v)
                        #policy_loss_v = sum(set_loss_policy_v)

                        #value_loss_v.backward()

                if flag_1:
                    set_traj.clear()
                    set_expert.clear()
                    if (mean_reward >= control_point)  and (iter>200):
                        control_point= control_point + 0.1
                        net_act_1.rise_length()
                        print("length: " + str(net_act_1.length))
                        print("control point "+str(control_point))
                        print("mean_reward "+str(mean_reward))
                        exp_buf, agent = exp_b(net_act_1,name, change, device, flag=True)

                        saves_path = 'C:/Users/corsa/Documents/ppo_checkpoints/'
                        idx = step_idx
                        check_1 = "checkpoint_actor-" + str(mean_reward) + "-%3d.data"
                        check_2 = "checkpoint_critic-" + str(mean_reward) + "-%3d.data"
                        torch.save(net_act_1.state_dict(), os.path.join(saves_path, check_1 % idx))
                        #torch.save(net_act_2.state_dict(), os.path.join(saves_path, check_1 % idx))
                        torch.save(net_crt.state_dict(), os.path.join(saves_path, check_2 % idx))
                        print("Saved model")
                    #break


