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
# importlib.reload(data)
path = 'C:/Users/silva/iCloudDrive/Docs/Ausbildung/QuantLibrary/MScQF_Thesis/9. APT Seminar/Returns_final.csv'
data = datapreprocessing.Preprocessing(path)

#%% Compute for each possible point in time within the data the n largest stocks based on market cap
data.compute_nlargest(100)

#%%
lookback_window = 600
rf              = 0.00
date            = 20181231
importlib.reload(opt)



#%%
permnos, market_weights, exp_returns, covars, returns_ahead = data.means_historical(date=date, lookback_window=lookback_window)
    
markowitz_pf      = opt.Markowitz(rf=rf, permnos=permnos, market_weights=market_weights, exp_returns=exp_returns, covars=covars)
markowitz_weights = markowitz_pf.solve_weights()
mc_weights        = markowitz_pf.solve_weights_mc()

#%%
plt.plot(mc_weights)
# plt.plot(markowitz_weights)
plt.show()

#%%
