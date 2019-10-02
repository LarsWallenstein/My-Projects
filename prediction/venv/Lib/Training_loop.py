from Model import Model_1
from Common import load_data, encode
import numpy as np
import os

import torch
import torch.optim as optim
import torch.nn.functional as F

LEARNING_RATE = 2e-4
NUMBER_OF_STEPS = 30
FORWARD_LENGTH = 4

ENVS_IN_BATCH = 512

if __name__ == "__main__":

    device = "cuda"
    #save_path = os.path.join("saves", "prediction-" + "move")
    save_path = "C:/Users/corsa/PycharmProjects/prediction/checkpoints"
    os.makedirs(save_path, exist_ok=True)

    net_act = Model_1(147, 2, p=0.1).to(device)
    net_act.load_state_dict(torch.load(os.path.join(save_path, "best_+0.153_63.dat")))
    net_act.reset()
    opt_net = optim.Adam(net_act.parameters(), lr=LEARNING_RATE)

    step_idx = 0
    best = 1000

    while True:
        step_idx+=1
        total_error = 0.0
        opt_net.zero_grad()
        nums = 0
        for z in range(ENVS_IN_BATCH):
            data, prices = load_data()
            if data.shape[0]>40:
                da, pr = encode(data=data, price=prices, forward=FORWARD_LENGTH, num_steps=NUMBER_OF_STEPS)
            net_act.reset()
            pr = torch.FloatTensor(pr).to(device)
            errors = []
            k=-1
            for obs in da:
                obs = torch.FloatTensor(obs.reshape((1,147))).to(device)
                prediction = net_act(obs)
                if(step_idx==1):
                    print("Prediction ",F.softmax(prediction[0]))
                    print("Right ", pr[k])
                k += 1 #number of observation in env
                if not(torch.isnan(prediction[0][0])or torch.isnan(prediction[0][1])):
                    #print(z)
                    #prediction = torch.FloatTensor([[0,0]]).to(device)
                    error = F.binary_cross_entropy_with_logits(prediction[0], pr[k])
                    total_error+=error.item()
                    errors.append(error)
                    nums+=1 #number of samples in training
            if(errors):
                loss_v = sum(errors)
                loss_v.backward()
        print(str(step_idx)+". Sum of errors: ", total_error/(nums*1.0))
        if(step_idx>20 and  (total_error/(nums*1.0)<best)):
            best = (total_error/(nums*1.0))
            name = "best_%+.3f_%d.dat" % (total_error/(nums*1.0), step_idx)
            fname = os.path.join(save_path, name)
            torch.save(net_act.state_dict(), fname)
        opt_net.step()