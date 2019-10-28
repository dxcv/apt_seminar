#%%
import numpy as np
import pandas as pd
import datetime
import logging
import datapreprocessing2
import optimizers2
import matplotlib.pyplot as plt
import importlib

#%%
# data      = pd.read_excel('Seminar_returns.xlsx')
# data.date = pd.to_datetime(data.date).dt.strftime('%Y%m%d')

#%%
importlib.reload(datapreprocessing2)
path    = 'C:/Users/silva/iCloudDrive/Docs/Ausbildung/QuantLibrary/MScQF_Thesis/9. APT Seminar/Returns_monthly2.csv'
# data    = datapreprocessing2.Preprocessing('Seminar_returns.xlsx')
data    = datapreprocessing2.Preprocessing(path)

#%% Compute for each possible point in time within the data the n largest stocks based on market cap
data.compute_nlargest(10)

#%%
lookback_window     = 25
rebal_period        = 1
start_date          = 20000831
end_date            = 20180831


#%%
tdates              = data.trading_dates[(data.trading_dates >= start_date) & (data.trading_dates <= end_date)]
horizon             = len(tdates)

if rebal_period > horizon:
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
importlib.reload(datapreprocessing2)
m_return    = np.array([])
ev_return   = np.array([])


for date in tdates:
    # Create subset of data where there are at least as many values as required by the lookback window
    # I.e. only consider assets that have at enough past returns for a given date    
    if date in rebal_dates:
        # SEMINAR RETURNS #

        # universe    = data.df.loc[(data.df.date <= date)][-lookback_window:].dropna(axis='columns')      
        # returns     = universe.drop(columns=['date', 'Cash CHF'])
        # rf          = universe['Cash CHF'].drop(columns=['date']).values[-2]
        # assets      = returns.columns
        # returns_t   = returns.values[-1]
        # returns_p   = returns[:-1]

        # CRSP RETURNS #
        assets, returns, _, _ = data.permnos_returns_caps_weights(date=date, lookback_window=lookback_window)
        returns     = np.asarray(returns).T
        returns_t   = returns[-1]
        returns_p   = returns[:-1]
        rf          = 0
        myopt       = optimizers2.MeanVariance(rf=rf, permnos=assets, returns=returns_p, rebal_period=rebal_period, mean_pred='Holt')

        W_m         = myopt.solve_tangency_weights()
        W_ev        = np.ones([len(assets)]) / len(assets)
    
    # universe        = data.df.loc[(data.df.date == date)]
    # returns_t       = np.squeeze(universe[universe.columns[universe.columns.isin(assets)]].values)
    # assets_t, returns, _, _ = data.permnos_returns_caps_weights(date=date, lookback_window=1)
    # print(assets_t in assets)

    W_m             = (W_m * (1+returns_t)) / np.dot(W_m, 1+returns_t)
    W_ev            = (W_ev * (1+returns_t)) / np.dot(W_ev, 1+returns_t)
    m_return        = np.append(m_return, W_m.dot(returns_t))
    ev_return       = np.append(ev_return, W_ev.dot(returns_t))

#%%

m_value     = 1+m_return
ev_value    = 1+ev_return

m_value = np.cumprod(m_value, axis=0)
ev_value = np.cumprod(ev_value, axis=0)

plt.plot(ev_value, label='EW')
plt.plot(m_value, label='MV')
legend = plt.legend(loc='upper left', shadow=False)



#%%
