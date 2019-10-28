#%%
import numpy as np
import pandas as pd
import datetime
import logging
import datapreprocessing2 as dp
import optimizers2
import matplotlib.pyplot as plt
import importlib

#%%
df = pd.read_excel('Seminar_returns.xlsx')
df.date = pd.to_datetime(df.date).dt.strftime('%Y%m%d')

#%%
lookback_window     = 24
rebal_period        = 3
start_date          = '20030131'
end_date            = '20190831'

dates               = df.date.unique()
start_position      = np.where(dates == start_date)[0].item(0)
end_position        = np.where(dates == end_date)[0].item(0)
obs_start           = dates[start_position-lookback_window+1]
tdates              = df.loc[(df.date >= start_date) & (df.date <= end_date),'date'].values
horizon             = len(tdates)

if rebal_period == 1:
    mod = divmod(horizon, rebal_period)[0]
elif rebal_period > horizon:
    rebal_period = 3
    logging.warning('Rebalancing period exceeds horizon. Value set to 3 months')
    mod = divmod(horizon, rebal_period)[0]
else:
    mod = divmod(horizon, rebal_period)[0]
    
counter     = 0
rebal_dates = []

for i in range(mod):
    rebal_date = tdates[counter]
    rebal_dates.append(rebal_date)
    counter += rebal_period

#%%

importlib.reload(optimizers2)
m_return    = np.array([])
ev_return   = np.array([])


for date in tdates:
    # Create subset of data where there are at least as many values as required by the lookback window
    # I.e. only consider assets that have at enough past returns for a given date    
    if date in rebal_dates:
        universe    = df.loc[(df.date <= date)][-lookback_window:].dropna(axis='columns')      
        returns     = universe.drop(columns=['date', 'Cash CHF'])
        rf          = universe['Cash CHF'].drop(columns=['date']).values[-2]
        returns_t   = returns.values[-1]
        returns_p   = returns[:-1]
        assets      = returns.columns

        myopt       = optimizers2.MeanVariance(rf=rf, permnos=assets, returns=returns_p, rebal_period=rebal_period, mean_pred='Holt')

        W_m         = myopt.solve_tangency_weights()
        W_ev        = np.ones([len(assets)]) / len(assets)
    
    universe        = df.loc[(df.date == date)]
    returns_t       = np.squeeze(universe[universe.columns[universe.columns.isin(assets)]].values)

    W_m             = (W_m * (1+returns_t)) / np.dot(W_m, 1+returns_t)
    W_ev            = (W_ev * (1+returns_t)) / np.dot(W_ev, 1+returns_t)
    m_return        = np.append(m_return, W_m.dot(returns_t))
    ev_return       = np.append(ev_return, W_ev.dot(returns_t))

#%%

m_value     = 1+m_return
ev_value    = 1+ev_return

m_value = np.cumprod(m_value, axis=0)
ev_value = np.cumprod(ev_value, axis=0)

plt.plot(ev_value)
plt.plot(m_value)


#%%
print(np.quantile(ev_return, 0.01))

#%%
plt.plot(m_return)
plt.plot(ev_return)

#%%
