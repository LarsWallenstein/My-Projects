import torch
from Common import *
from Model import *
from Env import *
import torch.nn.utils as nn_utils
import torch.nn.functional as F
import tensorboardX as tb
from ctypes import c_bool, c_char_p

GAMMA = 0.95
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01
BATCH_SIZE = 1

REWARD_STEPS = 1
CLIP_GRAD = 0.5

PROCESSES_COUNT = 1
NUM_ENVS = 64

s=False

REWARD_BOUND = 1000
CHECKPOINT_EVERY_STEP = 1000

import ptan
import torch.multiprocessing as mp


def make_env(net, i, flag):
    stk = StocksEnv(net=net)
    stk.set_name(str(i))
    if(flag):
        stk._state.set_length(net.length)
    return stk

def exp_b(net, name, change, device, flag=False):
    envs = [make_env(net, p, flag) for p in range(net.num_envs)]
    net.set_envs(envs, name, change)
    agent = ptan.agent.PolicyAgent(lambda x: net(x)[0], apply_softmax=True, device=device)
    exp_buf = []
    for l in range(NUM_ENVS):
        exp_source = ptan.experience.ExperienceSourceFirstLast(envs[l], agent, gamma=GAMMA, steps_count=REWARD_STEPS)
        exp_buf.append(exp_source)
    return exp_buf

TotalReward = collections.namedtuple('TotalReward', field_names='reward')

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    device = "cuda"
    PATH = 'C:/Users/corsa/Documents/Stocks_2/'
    NAME = "TRADE"
    name = "1"
    writer = tb.SummaryWriter(comment="-a3c-data_" + NAME + "_" + name)
    saves_path = PATH + '/' + str(len(os.listdir(path=PATH)) - 3)
    os.makedirs(saves_path, exist_ok=True)


    net = DDQN(126, 3, num_envs=NUM_ENVS).to(device)
    net.share_memory()
    #load_path = PATH + '/'+str(10)
    #net.load_state_dict(torch.load(os.path.join(load_path, "checkpoint-%3d.data" % 24)))
    #   net.reset()
    load_path = "C:/Users/corsa/Documents/Stocks_2/44/"
    net.load_state_dict(torch.load(os.path.join(load_path, "checkpoint-%3d.data" % 190)))
    #net.load_state_dict(torch.load(os.path.join(load_path, "checkpoint1.606726802367143-138")))
    net.reset()
    optimizer = optim.Adadelta(net.parameters(), lr=LEARNING_RATE) # Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3)
    ####################################################################_____________PARAMETERS________________###########################################################################################################
    train_queue = [None]
    reward_queue = [None]
    batch = []
    step_idx = 0
    lol = 0
    i = 1
    iter = 0
    rew=-200
    idx=-1
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

    try:
        with RewardTracker(writer, stop_reward=REWARD_BOUND, num_of_envs=NUM_ENVS) as tracker:
            with ptan.common.utils.TBMeanTracker(writer, batch_size=100) as tb_tracker:
                exp_buf=exp_b(net,name,change,device)
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
                            reward_queue[0]=TotalReward(reward=np.mean(new_rewards))

                        train_queue[0]=exp
                        train_entry = train_queue[0]
                        if reward_queue[0] is not None:
                            condition, rew = tracker.reward(reward_queue[0].reward, step_idx)
                            if condition:
                                i = -1
                                break
                            # if (tracker.hist-tracker.last_mean)>0.5 and not(idx==-1):
                            #    net.load_state_dict(torch.load(os.path.join(saves_path, "checkpoint-%3d.data" % idx)))
                            #   net.reset()
                            if tracker.save and op:
                                op = False
                                idx = step_idx // CHECKPOINT_EVERY_STEP
                                oh = "checkpoint"+str(rew)+"-%3d.data"
                                torch.save(net.state_dict(), os.path.join(saves_path,oh  % idx))
                                print("Saved model")
                            # elif idx>0:
                            #   net.load_state_dict(torch.load(os.path.join(saves_path, "checkpoint-%3d.data" % idx)))
                            reward_queue[0] = None
                            top = True
                            iter = iter + 1
                            # continue

                        step_idx += 1
                        batch.append(train_entry)
                        # if len(batch) < BATCH_SIZE:
                        #   continue
                        if (top):
                            states = []
                            actions = []
                            vals = []
                            for z in range(len(batch)):
                                states_v, actions_t, vals_ref_v = unpack_batch([batch[z]], net,
                                                                               last_val_gamma=GAMMA ** REWARD_STEPS,
                                                                               device=device)
                                states.append(states_v)
                                actions.append(actions_t)
                                vals.append(vals_ref_v)
                            batch.clear()
                            net.reset()
                            assert len(states) == len(actions) and len(actions) == len(vals)
                            losses=[]
                            for z in range(len(states)):
                                logits_v, value_v = net(states[z])
                                if True:
                                    loss_value_v = F.mse_loss(value_v.squeeze(-1), vals[z])

                                    log_prob_v = F.log_softmax(logits_v, dim=1)
                                    adv_v = vals[z] - value_v.detach()
                                    log_prob_actions_v = adv_v * log_prob_v[range(BATCH_SIZE), actions[z]]
                                    loss_policy_v = -log_prob_actions_v.mean()

                                    prob_v = F.softmax(logits_v, dim=1)
                                    entropy_loss_v = ENTROPY_BETA * (prob_v * log_prob_v).sum(dim=1).mean()

                                    loss_v = (entropy_loss_v + loss_value_v + loss_policy_v) / (NUM_ENVS)
                                    losses.append(loss_v)
                                if z == (len(states) - 1):
                                    loss = sum(losses)
                                    loss.backward()

                                    tb_tracker.track("advantage", adv_v, step_idx)
                                    tb_tracker.track("values", value_v, step_idx)
                                    tb_tracker.track("batch_rewards", vals_ref_v, step_idx)
                                    tb_tracker.track("loss_entropy", entropy_loss_v, step_idx)
                                    tb_tracker.track("loss_policy", loss_policy_v, step_idx)
                                    tb_tracker.track("loss_value", loss_value_v, step_idx)
                                    tb_tracker.track("loss_total", loss_v, step_idx)
                                    """
                                if(z==len(states)-1):
                                    loss_ = sum(losses)
                                    loss_.backward()"""
                            if (rew - max_obtained_reward >= 0.2):
                                max_obtained_reward = rew
                                net.rise_length()
                                print("length: "+str(net.length))
                                exp_buf = exp_b(net, name, change, device, flag=True)
                            if (iter == NUM_ENVS):
                                iter = 0
                                nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
                                op = True
                                optimizer.step()
                                optimizer.zero_grad()
                                print("Step")
                            if(step_idx-idx==1000 and not s):
                                optimizer = optim.Adadelta(net.parameters(), lr=LEARNING_RATE*2)
                                s=True
                            if (step_idx - idx == 1000 and s):
                                optimizer = optim.Adadelta(net.parameters(), lr=LEARNING_RATE/2.0)
                                s=False
                        # if step_idx % CHECKPOINT_EVERY_STEP == 0:
                        #   idx = step_idx // CHECKPOINT_EVERY_STEP
                        #   torch.save(net.state_dict(), os.path.join(saves_path, "checkpoint-%3d.data" % idx))
                        if(top):
                            top=False
                            break
    finally:
        pass