import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import os

import random

def generate_data():
    mas = np.zeros((1,5))
    long_or_short = random.randint(0,2)
    difference = random.gauss(0.0,0.0045)
    if(long_or_short!=2):
        mas[0][long_or_short]=1.0
        mas[0][2] = difference
    long = random.random()
    short = 1.0-long
    mas[0][3] = long
    mas[0][4] = short
    var_1 = long_or_short
    if(long>=0.8):
        var_2 = 0
    elif(long>=0.7):
        var_2 = 1
    elif(long>0.3):
        var_2 = 2
    else:
        var_2 = 3
    if(long_or_short!=2):
        if(difference>=0.01):
            var_3 = 0
        elif(difference>=-0.01):
            var_3 = 1
        else:
            var_3 = 2
    else:
        var_3 = -1

    #if long
    if(var_1==0 and var_2 == 0 and var_3 == 0):
        ans = np.array([0.0,0.7,0.3])
    elif(var_1==0 and var_2 == 0 and var_3 == 1):
        ans = np.array([0.0, 0.8, 0.2])
    elif (var_1 == 0 and var_2 == 0 and var_3 == 2):
        ans = np.array([0.0, 0.4, 0.6])

    elif(var_1 == 0 and var_2 == 1 and var_3 == 0):
        ans = np.array([0.0, 0.55, 0.45])
    elif (var_1 == 0 and var_2 == 1 and var_3 == 1):
        ans = np.array([0.0, 0.7, 0.3])
    elif (var_1 == 0 and var_2 == 1 and var_3 == 2):
        ans = np.array([0.0, 0.3, 0.7])

    elif (var_1 == 0 and var_2 == 2 and var_3 == 0):
        ans = np.array([0.0, 0.2, 0.8])
    elif (var_1 == 0 and var_2 == 2 and var_3 == 1):
        ans = np.array([0.0, 0.25, 0.75])
    elif (var_1 == 0 and var_2 == 2 and var_3 == 2):
        ans = np.array([0.0, 0.1, 0.9])

    elif (var_1 == 0 and var_2 == 3 and var_3 == 0):
        ans = np.array([0.0, 0.0, 1.0])
    elif (var_1 == 0 and var_2 == 3 and var_3 == 1):
        ans = np.array([0.0, 0.0, 1.0])
    elif (var_1 == 0 and var_2 == 3 and var_3 == 2):
        ans = np.array([0.0, 0.0, 1.0])

    #if short
    elif(var_1 == 1 and var_2 == 0 and var_3 == 0):
        ans = np.array([1.0, 0.0, 0.0])
    elif (var_1 == 1 and var_2 == 0 and var_3 == 1):
        ans = np.array([1.0, 0.0, 0.0])
    elif (var_1 == 1 and var_2 == 0 and var_3 == 2):
        ans = np.array([1.0, 0.0, 0.0])

    elif (var_1 == 1 and var_2 == 1 and var_3 == 0):
        ans = np.array([0.9, 0.1, 0.0])
    elif (var_1 == 1 and var_2 == 1 and var_3 == 1):
        ans = np.array([0.75, 0.25, 0.0])
    elif (var_1 == 1 and var_2 == 1 and var_3 == 2):
        ans = np.array([1.0, 0.0, 0.0])

    elif (var_1 == 1 and var_2 == 2 and var_3 == 0):
        ans = np.array([0.8, 0.2, 0.0])
    elif (var_1 == 1 and var_2 == 2 and var_3 == 1):
        ans = np.array([0.75, 0.25, 0.0])
    elif (var_1 == 1 and var_2 == 2 and var_3 == 2):
        ans = np.array([0.75, 0.25, 0.0])

    elif (var_1 == 1 and var_2 == 3 and var_3 == 0):
        ans = np.array([0.2, 0.8, 0.0])
    elif (var_1 == 1 and var_2 == 3 and var_3 == 1):
        ans = np.array([0.1, 0.9, 0.0])
    elif (var_1 == 1 and var_2 == 3 and var_3 == 2):
        ans = np.array([0.45, 0.55, 0.0])

    #No pos
    elif(var_1 == 2 and var_2 == 0):
        ans = np.array([1.0, 0.0, 0.0])

    elif (var_1 == 2 and var_2 == 1):
        ans = np.array([0.8, 0.2, 0.0])

    elif (var_1 == 2 and var_2 == 2):
        ans = np.array([0.15, 0.7, 0.15])

    elif (var_1 == 2 and var_2 == 3):
        ans = np.array([0.0, 0.0, 1.0])

    return torch.FloatTensor(mas).to("cuda"), torch.FloatTensor(ans.reshape((1,3))).to("cuda")

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.linear_1 = nn.Linear(5,10)
        self.linear_2 = nn.Linear(10,3)

    def forward(self, var):
        #print(var)
        out = self.linear_1(var)
        out = self.linear_2(out)
        return out

LEARNING_RATE = 0.001

if __name__ == "__main__":
    net = model().to("cuda")
    opt = optim.Adam(net.parameters(),lr=LEARNING_RATE)

    save_path = "C:/Users/corsa/PycharmProjects/prediction/checkpoints_2"
    os.makedirs(save_path, exist_ok=True)
    best = 0.7

    while True:
        error =0.0
        errors = []
        for i in range(1024):
            data, out = generate_data()
            out_net = net(data)
            loss_v = F.binary_cross_entropy_with_logits(out_net[0], out[0])
            errors.append(errors)
            error += loss_v.item()
            loss_v.backward()
        print(error/1024.0)
        opt.step()
        for i in range(1):
            data, out = generate_data()
            out_net = net(data)
            print(out[0], F.softmax(out_net[0]))
        if(error/1024.0-best<-0.01):
            best = error/1024.0-0.01
            name = "best_%+.3f.dat" % (error/1024.0)
            fname = os.path.join(save_path, name)
            torch.save(net.state_dict(), fname)
