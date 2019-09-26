#%%
import numpy as np
import pandas as pd
import json

import scipy.optimize
import scipy as sp
import scipy.stats as stat
import matplotlib.pyplot as plt

import importlib
import datapreprocessing
import data
import optimizers as opt

import time


#%% Load the data 
#path = Path('C:\Users\silva\iCloudDrive\Docs\Ausbildung\QuantLibrary\MScQF_Thesis\9. APT Seminar\Returns_Short.csv')
importlib.reload(datapreprocessing)
importlib.reload(data)
path = 'C:/Users/silva/iCloudDrive/Docs/Ausbildung/QuantLibrary/MScQF_Thesis/9. APT Seminar/Returns_final.csv'
data1 = datapreprocessing.Preprocessing(path)
data2 = data.Data(path)

#%% Compute for each possible point in time within the data the n largest stocks based on market cap
data.compute_nlargest(20)



#%%
lookback_window = 740
rf              = -0.0075
max_length      = len(data.trading_dates)-lookback_window
print(max_length)

dates           = data.trading_dates[-max_length:]
print(len(dates))

#%%
importlib.reload(opt)
markowitz_returns = np.array([])
bl_returns = np.array([])
market_returns = np.array([])

start_time = time.time()

for date in dates:
    #_, market_weights, _, _, _,  returns_ahead = data_set.means_historical(date=date, lookback_window=lookback_window)
    permnos, market_weights, exp_returns, covars, returns_ahead = data.means_historical(date=date, lookback_window=lookback_window)
    markowitz_pf = opt.Markowitz(rf=rf, permnos=permnos, market_weights=market_weights, exp_returns=exp_returns, covars=covars)
    bl_pf = opt.Markowitz(rf=rf, permnos=permnos, market_weights=market_weights, exp_returns=exp_returns, covars=covars)
    
    markowitz_weights = markowitz_pf.solve_weights()
    bl_weights = bl_pf.solve_weights()

    markowitz_returns = np.append(markowitz_returns, np.array(markowitz_weights).dot(np.array(returns_ahead)))
    bl_returns = np.append(bl_returns, np.array(bl_weights).dot(np.array(returns_ahead)))
    market_returns = np.append(market_returns, np.array(market_weights).dot(np.array(returns_ahead)))

print("--- %s seconds ---" % (time.time() - start_time))

#%%
plt.plot(market_returns)
plt.plot(bl_returns)
#plt.plot(market_returns)
plt.show()

#%%
markowitz_value = markowitz_returns + 1
markowitz_value = np.cumprod(markowitz_value, axis=0)
market_value = market_returns + 1
market_value = np.cumprod(market_value, axis=0)
plt.plot(markowitz_value)
plt.plot(market_value)
plt.show()

#%%
