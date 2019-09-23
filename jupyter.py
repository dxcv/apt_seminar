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

#%% Compute for each possible point in time within the data the n largest stocks based on market cap
data_set.compute_nlargest(20)



#%%
lookback_window = 700
rf              = -0.0075
max_length      = len(data_set.trading_dates)-lookback_window
dates           = data_set.trading_dates[-max_length:]

#%%
portfolio_returns = np.array([])

for date in dates:
    permnos, market_weights, exp_returns, covars, _,  returns_ahead = data_set.means_historical(date=date, lookback_window=lookback_window)
    #markowitz_pf = opt.Markowitz(rf=rf, permnos=permnos, market_weights=market_weights, exp_returns=exp_returns, covars=covars)
    #portfolio_weights = markowitz_pf.solve_weights()
    #portfolio_returns = np.append(portfolio_returns, np.transpose(portfolio_weights)*returns_ahead)
    print(permnos)

#%%
