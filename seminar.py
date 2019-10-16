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
lookback_window     = 48
rebal_period        = 12
start_date          = '19980131'
end_date            = '20190131'

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
import importlib
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

        myopt       = optimizers2.Markowitz(rf=rf, permnos=assets, returns=returns_p, rebal_period=rebal_period)
        
        W_m         = myopt.solve_weights()
        W_ev        = np.ones([len(assets)]) / len(assets)
    
    universe        = df.loc[(df.date == date)]
    returns_t       = np.squeeze(universe[universe.columns[universe.columns.isin(assets)]].values)
    W_m             = (W_m * (1+returns_t)) / np.dot(W_m, 1+returns_t)
    W_ev            = (W_ev * (1+returns_t)) / np.dot(W_ev, 1+returns_t)
    m_return        = np.append(m_return, W_m.dot(returns_t))
    ev_return       = np.append(ev_return, W_ev.dot(returns_t))


#%%
import matplotlib.pyplot as plt
m_value     = 1+m_return
ev_value    = 1+ev_return

m_value = np.cumprod(m_value, axis=0)
ev_value = np.cumprod(ev_value, axis=0)

plt.plot(ev_value)
plt.plot(m_value)


# TIME-SERIES FORECASTING

#%%
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import matplotlib.pyplot as plt

#%%
universe = df.loc[(df.date <= '20140530')][-60:].dropna(axis='columns').drop(columns=['date', 'Cash CHF'])

returns = np.matrix(universe)
_, cols = np.shape(returns)

for asset in range(cols):
    returns     = (1 + universe[asset]).reset_index(drop=True).cumprod()
    model       = Holt(returns, exponential=True).fit(smoothing_level=0.8, smoothing_slope=0.2, optimized=False)
    pred        = model.forecast(3).values[-1]/returns.values[-1]-1

#%%
universe    = df.loc[(df.date <= '20140530')][-60:].dropna(axis='columns').drop(columns=['date', 'Cash CHF'])

for asset in universe.columns:
    returns     = (1 + universe[asset]).reset_index(drop=True).cumprod()
    train, test = returns.head(57), returns.tail(3)
    model       = Holt(train, exponential=True).fit(smoothing_level=0.8, smoothing_slope=0.2, optimized=False)
    pred        = model.forecast(3).values[-1]/train.values[-1]-1

    # print('predicted return:', pred.values[-1]/train.values[-1]-1)
    # print('realized return:', returns.values[-1]/train.values[-1]-1)



#%%

#%%


# plt.plot(train.index, train, label='Train')
# plt.plot(test.index, test, label='Test')
# plt.plot(pred.index, pred, label='Holt-Winters')
# plt.legend(loc='best')





#%%
