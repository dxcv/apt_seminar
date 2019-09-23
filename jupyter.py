#%%
import numpy as np
import pandas as pd
import json

import scipy.optimize
import scipy as sp
import scipy.stats as stat
import matplotlib.pyplot as plt

# My libraries
from Data import Data
from Optimizers import Markowitz, BlackLitterman

#%% Load the data 
#path = Path('C:\Users\silva\iCloudDrive\Docs\Ausbildung\QuantLibrary\MScQF_Thesis\9. APT Seminar\Returns_Short.csv')
path = 'C:/Users/silva/iCloudDrive/Docs/Ausbildung/QuantLibrary/MScQF_Thesis/9. APT Seminar/Returns_Short.csv'
data = Data(path)

#%% Compute for each possible point in time within the data the n largest stocks based on market cap
data.compute_nlargest(20)

#%%
lookback_window = 600
rf              = -0.075
max_length      = len(data.trading_dates)-lookback_window
dates           = data.trading_dates[-max_length:]

#%% 
permnos, market_weights, exp_returns, covars, _ = data.means_historical(date=20181231, lookback_window=500)

#%%
markowitz_pf    = Markowitz(rf=rf, permnos=permnos, market_weights=market_weights, exp_returns=exp_returns, covars=covars)
bl_pf           = BlackLitterman(rf=rf, permnos=permnos, market_weights=market_weights, exp_returns=exp_returns, covars=covars)

#%%
markowitz_pf.display_assets()


#%%
print('hello')