import numpy as np
import pandas as pd

def ema(x, y):  # Не происходит никаких сокращений данных
    alpha = 2 / (y + 1)
    summ = []
    summ.append(0.0)
    for i in range(len(x)):
        summ.append(x[i] * alpha + (1 - alpha) * summ[-1])
    return np.around(np.array(summ[1:]), 3)

def sma(x, y):  # работаетб но первые y значений не являются правильными
    summ = []
    summ.append(x[0])
    for i in range(y - 1):
        summ.append((sum(x[:i + 1]) + x[i + 1]) / (i + 2))
    for j in range(y, len(x)):
        summ.append(sum(x[j - y + 1:j + 1]) / (1.0 * y))
    return np.around(np.array(summ), 3)

def wt_lb(data, N1=6,N2=4, I1=3,I2=2):
    N1 = N1  # Channel length
    N2 = N2  # Average Length

    I1 = I1
    I2 = I2

    #obLevel1 = 1000  # OverBought level
    # obLevel2 = 53
    #osLevel1 = -1000  # OverSold level
    # osLevel2 = -53

    #w1 = 105
    #w2 = 90

    ap = (data['HaHigh'] + data['HaClose']) / 2.0
    esa = ema(ap, N1)
    d = ema(abs(ap - esa), N1)
    ci = (ap - esa) / (0.015 * d)
    tci = ema(ci, N2)

    wt1 = sma(tci, I1)
    wt2 = pd.Series(sma(wt1, I2), name="Wt2")
    wt1 = pd.Series(wt1, name="Wt1")

    return pd.concat([data, wt1, wt2], axis=1)

def weis_wave(x, length=5):
    up_mas = [0]
    down_mas = [0]
    mov_mas = [0]
    wave_mas = [0]
    mas_vol = [0]
    trend_mas = [0]
    #mov = 0
    for i in range(1, x.shape[0]):
        if (x['HaClose'].iloc[i - 1] < x['HaClose'].iloc[i]):
            mov = 1
        elif (x['HaClose'].iloc[i - 1] > x['HaClose'].iloc[i]):
            mov = -1
        else:
            mov = 0
        if (mov != 0 and mov != mov_mas[-1]):
            trend = mov
        else:
            trend = trend_mas[-1]
        is_trending = (x['HaClose'].iloc[i] > x['HaClose'].iloc[i - length:i]).all() or (
                    x['HaClose'].iloc[i] < x['HaClose'].iloc[i - length:i]).all()
        wave = trend if ((trend != wave_mas[-1]) and is_trending) else wave_mas[-1]
        vol = x['Volume'].iloc[i] + mas_vol[-1] if wave == wave_mas[-1] else x['Volume'].iloc[i]
        up = vol if wave == 1 else 0
        if (wave == 1):
            dn = 0
        elif (wave == -1):
            dn = -vol
        else:
            dn = 0
        up_mas.append(up)
        down_mas.append(dn)
        mov_mas.append(mov)
        wave_mas.append(wave)
        mas_vol.append(vol)
        trend_mas.append(trend)
    up_mas = pd.Series(up_mas, name="Up")
    down_mas = pd.Series(down_mas, name="Down")
    return pd.concat([x, up_mas, down_mas], axis=1)

def heikin_ashi(x):
    haClose = (x['Open'] + x['Close'] + x['Low'] + x['High']) / 4.0
    haOpen = [0] * x.shape[0]
    haHigh = [0] * x.shape[0]
    haLow = [0] * x.shape[0]
    color = [0] * x.shape[0]
    color_2 = [0] * x.shape[0]
    haOpen[0] = x['Open'].iloc[0]
    haHigh[0] = x['High'].iloc[0]
    haLow[0] = x['Low'].iloc[0]
    color[0] = 1 if x['Close'].iloc[0] > x['Open'].iloc[0] else -1
    color_2[0] = 1 if x['Close'].iloc[0] > x['Open'].iloc[0] else -1
    for i in range(1, x.shape[0]):
        haOpen[i] = (haOpen[i - 1] + haClose.iloc[i - 1]) / 2.0
        haHigh[i] = max(x['High'].iloc[i], haOpen[i], haClose.iloc[i])
        haLow[i] = min(x['Low'].iloc[i], haOpen[i], haClose.iloc[i])
        color_2[i] = 1 if x['Close'].iloc[i] > x['Open'].iloc[i] else -1
        if (haClose.iloc[i] > haOpen[i]):
            color[i] = 1
        else:
            color[i] = -1
    haOpen = pd.Series(haOpen, name='HaOpen')
    haHigh = pd.Series(haHigh, name='HaHigh')
    haLow = pd.Series(haLow, name='HaLow')
    color = pd.Series(color, name='HaColor')
    color_2 = pd.Series(color_2, name='Color')
    haClose.name = 'HaClose'
    return pd.concat([x, haOpen, haHigh, haLow, haClose, color, color_2], axis=1)

if __name__ == "__main__":
    PATH = "C:/Users/corsa/Documents/Stocks/5 Min/ETFs/tvix.us.txt"
    data = pd.read_csv(PATH, sep=',').round(5)
    data = weis_wave(wt_lb(heikin_ashi(data)))
    best_reward = None
    N1_best = 0
    N2_best = 0
    I1_best = 0
    I2_best = 0
    length_best = 0
    for N1 in range(2,12):
        #print(N1)
        for N2 in range(2,12):
            print(N2)
            for I1 in range(1,12):
                for I2 in range(1,12):
                    for length in range(1,12):
                        data = pd.read_csv(PATH, sep=',').round(5)
                        data = weis_wave(wt_lb(heikin_ashi(data), N1=N1, N2=N2,I1=I1,I2=I2),length=length)
                        buy_sig = False
                        sell_sig = False
                        open_pos = -100.0
                        reward = 1.0
                        minimum = 1.0
                        for i in range(20,data.shape[0]):
                            if(data.loc[i,'Up']>0 and data.loc[i,'Wt1']>data.loc[i,'Wt2']):
                                if(sell_sig):
                                    sell_sig = False
                                    reward *=0.999
                                    reward *= (1 + (open_pos - data.loc[i, 'Close']) / (1.0 * data.loc[i, 'Close']))
                                    if(reward<minimum):
                                        minimum=reward
                                if(not buy_sig):
                                    buy_sig = True
                                    open_pos = data.loc[i, 'Close']
                                    reward *= 0.999
                                    if (reward < minimum):
                                        minimum = reward
                            if(data.loc[i,'Down']<0 and data.loc[i,'Wt1']<data.loc[i,'Wt2']):
                                if(buy_sig):
                                    buy_sig=False
                                    reward*=0.999
                                    reward *= (1.0+(data.loc[i,'Close']-open_pos)/(1.0*open_pos))
                                    if (reward < minimum):
                                        minimum = reward
                                if(not sell_sig):
                                    sell_sig = True
                                    reward*=0.999
                                    open_pos = data.loc[i, 'Close']
                                    if (reward < minimum):
                                        minimum = reward
                            #print(reward)
                            if (i == data.shape[0] - 1):
                                if (buy_sig):
                                    reward *= (data.loc[i, 'Close'] - open_pos)/(1.0*open_pos)
                                if (sell_sig):
                                    reward *= (open_pos - data.loc[i, 'Close'])/(1.0*data.loc[i, 'Close'])
                                if (reward < minimum):
                                    minimum = reward
                            """
                            if(data.loc[i,'Up']>0 and data.loc[i-1,'Down']<0 and data.loc[i,'Wt1']>data.loc[i,'Wt2']):
                                if(sell_sig):
                                    sell_sig = False
                                    reward*=(1+(open_pos-data.loc[i,'Close'])/(1.0*data.loc[i,'Close']))
                                if(not buy_sig):
                                    buy_sig=True
                                    open_pos = data.loc[i,'Close']
                                    reward*=0.999
                            elif(data.loc[i-1, 'Up'] > 0 and data.loc[i, 'Down'] < 0 and data.loc[i, 'Wt1'] < data.loc[i, 'Wt2']):
                                if(buy_sig):
                                    buy_sig=False
                                    reward*=(1.0+(data.loc[i,'Close']-open_pos)/(1.0*open_pos))
                                if(not sell_sig):
                                    sell_sig=True
                                    open_pos = data.loc[i, 'Close']
                                    reward*=0.999
                            if(i==data.shape[0]-1):
                                if(buy_sig):
                                    reward+=(data.loc[i,'Close']-open_pos)
                                if(sell_sig):
                                    reward += (open_pos-data.loc[i, 'Close'])
                            """
                        #print(reward)
                        if(best_reward is None):
                            best_reward=reward
                            N1_best = N1
                            N2_best = N2
                            I1_best = I1
                            I2_best = I2
                            length_best = length
                            print("best reward ", best_reward, "N1: ", N1_best, "N2: ", N2_best, "I1: ", I1_best,"I2: ", I2_best, "length: ", length_best, 'min', minimum)
                        if(reward>best_reward):
                            best_reward = reward
                            N1_best = N1
                            N2_best = N2
                            I1_best = I1
                            I2_best = I2
                            length_best = length
                            print("best reward ", best_reward,"N1: ",N1_best, "N2: ", N2_best,"I1: ",I1_best,"I2: ",I2_best,"length: ",length_best,'min' ,minimum)

