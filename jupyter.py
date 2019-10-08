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
importlib.reload(datapreprocessing)
path = 'C:/Users/silva/iCloudDrive/Docs/Ausbildung/QuantLibrary/MScQF_Thesis/9. APT Seminar/Returns_final.csv'
data = datapreprocessing.Preprocessing(path)

#%% Compute for each possible point in time within the data the n largest stocks based on market cap
data.compute_nlargest(20)

#%%
lookback_window = 650
rf              = 0.00
max_length      = len(data.trading_dates)-lookback_window
print(max_length)

dates           = data.trading_dates[-max_length:]
print(len(dates))


#%%
importlib.reload(opt)

markowitz_returns   = np.array([])
market_returns      = np.array([])
ew_returns          = np.array([])

start_time = time.time()

for date in dates:
    permnos, market_weights, exp_returns, covars, returns_ahead = data.means_historical(date=date, lookback_window=lookback_window)
    
    markowitz_pf      = opt.Markowitz(rf=rf, permnos=permnos, market_weights=market_weights, exp_returns=exp_returns, covars=covars)
    
    n_assets          = len(market_weights)

    markowitz_weights = markowitz_pf.solve_weights()
    equal_weights     = np.ones([n_assets]) / n_assets 

    markowitz_returns = np.append(markowitz_returns, np.array(markowitz_weights).dot(np.array(returns_ahead)))
    market_returns    = np.append(market_returns, np.array(market_weights).dot(np.array(returns_ahead)))
    ew_returns        = np.append(ew_returns, np.array(equal_weights).dot(np.array(returns_ahead)))

print("--- %s seconds ---" % (time.time() - start_time))

#%%
plt.hist(markowitz_returns)
plt.hist(market_returns)
plt.hist(ew_returns)
plt.show()

#%%
markowitz_value = markowitz_returns + 1
markowitz_value = np.cumprod(markowitz_value, axis=0)
market_value = market_returns + 1
market_value = np.cumprod(market_value, axis=0)
ew_value = ew_returns+1
ew_value = np.cumprod(ew_value, axis=0)

plt.plot(markowitz_value, label='Markowitz')
plt.plot(market_value, label='Market-Value')
plt.plot(ew_value, label='Equally-weighted')
legend = plt.legend(loc='upper left', shadow=False)

plt.show()

#%%
std_markowitz = np.std(markowitz_returns)
std_market = np.std(market_returns)
std_mv = np.std(ew_returns)

print('Markowitz Std:', std_markowitz)
print('MV Std:', std_mv)
print('Market Std:', std_market)

#%%

print(dates)

#%% FAMA FRENCH
path = 'C:/Users/silva/iCloudDrive/Docs/Ausbildung/QuantLibrary/MScQF_Thesis/9. APT Seminar/FF_5_daily.csv'
ff = pd.read_csv(path)


#%%
ff.head()


#%%
