import random
import pandas as pd
import numpy as np
import os

def csv_to_txt(string):
    subStrOld = "csv"
    subStrNew = "txt"
    lenStrOld = len(subStrOld)

    while string.find(subStrOld) > 0:
        i = string.find(subStrOld)
        string = string[:i] + subStrNew + string[i + lenStrOld:]
    return string

def load_data():  # 1. data, 2.prices
    time_frame = str(random.randint(2,3))
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
        if not np.isnan(d1).any():
            break
    # return np.array(pd.read_csv(PATH,sep=',').round(5)), pd.read_csv(PATH_2,sep=',').round(5)
    # print(stocks[num], time_frame, der)
    return d1, pd.read_csv(PATH_2+ '/' + csv_to_txt(stocks[num]), sep=',').round(5)# + '/' + csv_to_txt("tvix.us.csv"), sep=',').round(5)

def encode(data, price, forward, num_steps):
    offset = random.randint(2,data.shape[0]-(num_steps+forward+1))
    da = np.ndarray(shape=(num_steps,49*3), dtype=np.float32)
    pr = np.ndarray(shape=(num_steps,2), dtype=np.float32)
    for i in range(num_steps):
        kek = np.array([])
        for k in range(forward-1):
            kek = np.concatenate((kek, data[offset-k+i]), axis=None)
        da[i]=kek
        rise=1.0
        for j in range(forward):
            rise*=((price.loc[offset+j,'Close']-price.loc[offset,'Close'])/(price.loc[offset, 'Close']*1.0)+1)
        if(rise>1):
            pr[i] = np.array([1,0])
        else:
            pr[i] = np.array([0,1])
    return da, pr