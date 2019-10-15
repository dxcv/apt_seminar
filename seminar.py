#%%
import numpy as np
import pandas as pd
import datetime
import logging
import datapreprocessing2 as dp
import optimizers2


#%%
df = pd.read_excel('Seminar_returns.xlsx')
df.date = pd.to_datetime(df.date).dt.strftime('%Y%m%d')


#%%
lookback_window     = 24
rebal_period        = 12
start_date          = '19870131'
end_date            = '19880131'

dates               = df.date.unique()
start_position      = np.where(dates == start_date)[0].item(0)
end_position        = np.where(dates == end_date)[0].item(0)
obs_start           = dates[start_position-lookback_window+1]
tdates              = df.loc[(df.date >= start_date) & (df.date <= end_date),'date'].values
odates              = df.loc[(df.date >= obs_start) & (df.date <= end_date),'date'].values
horizon             = len(tdates)

if rebal_period == 1:
    mod = divmod(horizon, rebal_period)[0]
elif rebal_period > horizon:
    rebal_period = 3
    logging.warning('Rebalancing period exceeds horizon. Value set to 3 months')
    mod = divmod(horizon, rebal_period)[0]
else:
    mod = divmod(horizon, rebal_period)[0]
    
#%%
counter     = 0
rebal_dates = []

for i in range(mod):
    rebal_date = tdates[counter]
    rebal_dates.append(rebal_date)
    counter += rebal_period

#%%
port_return = np.array([])

for date in tdates:
    # Create subset of data where there are at least as many values as required by the lookback window
    # I.e. only consider assets that have at enough past returns for a given date    
    if date in rebal_dates:
        universe    = df.loc[(df.date <= date)][-lookback_window:].dropna(axis='columns')
        returns     = universe.drop(columns=['date', 'Cash CHF'])
        rf          = universe['Cash CHF'].drop(columns=['date'])
        assets      = returns.columns
        returns_t   = np.squeeze(returns.values[-1])
        returns_p   = returns[:-1]
        # myopt       = optimizers2.Optimizer(rf=rf, permnos=assets, returns=returns_p)
        # cov         = myopt.C
        
        # Strategy
        W_t         = np.ones([len(assets)]) / len(assets)
    
    universe        = df.loc[(df.date == date)]
    returns_t       = np.squeeze(universe[universe.columns[universe.columns.isin(assets)]].values)
    W_t             = (W_t * (1+returns_t)) / np.dot(W_t, 1+returns_t)
    port_return     = np.append(port_return, W_t.dot(returns_t))


#%%
import matplotlib.pyplot as plt
port_value = 1+port_return
port_value = np.cumprod(port_value, axis=0)

plt.plot(port_value)

#%%
print(pd.__version__)

#%%
