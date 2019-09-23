#%%
import numpy as np
import pandas as pd
import json

import scipy.optimize
import scipy as sp
import scipy.stats as stat
import matplotlib.pyplot as plt

import importlib
import data
import optimizers as opt


#%% Load the data 
#path = Path('C:\Users\silva\iCloudDrive\Docs\Ausbildung\QuantLibrary\MScQF_Thesis\9. APT Seminar\Returns_Short.csv')
importlib.reload(data)
path = 'C:/Users/silva/iCloudDrive/Docs/Ausbildung/QuantLibrary/MScQF_Thesis/9. APT Seminar/Returns_final.csv'
data_set = data.Data(path)

#%%
data_set.df.head(425)

#%% Compute for each possible point in time within the data the n largest stocks based on market cap
data_set.compute_nlargest(20)



#%%
lookback_window = 700
rf              = -0.0075
max_length      = len(data_set.trading_dates)-lookback_window
dates           = data_set.trading_dates[-max_length:]
print(dates)

#%%
returns = data_set.permnos_returns_caps_weights(date=20181231, lookback_window=lookback_window)[1]

#%% 
permnos, market_weights, exp_returns, covars, _ = data_set.means_historical(date=20181231, lookback_window=500)

#%%
importlib.reload(opt)
markowitz_pf    = opt.Markowitz(rf=rf, permnos=permnos, market_weights=market_weights, exp_returns=exp_returns, covars=covars)
bl_pf           = opt.BlackLitterman(rf=rf, permnos=permnos, market_weights=market_weights, exp_returns=exp_returns, covars=covars)

#%%

for date in dates:
    
    print(date)

#%%
