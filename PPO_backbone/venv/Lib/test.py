import numpy as np
import pandas as pd

if __name__ == '__main__':
    PATH = 'C:/Users/corsa/Documents/Stocks_2_new/1 Hour/ETFs'
    d1 = pd.read_csv(PATH + '/' + "adru.us.csv", sep=',').round(5)
    print(d1.mean(axis=0))