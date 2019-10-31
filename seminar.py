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
df          = pd.read_excel('Seminar_returns.xlsx', sheet_name='time series')
df_w        = pd.read_excel('Seminar_returns.xlsx', sheet_name='market caps')
df.date     = pd.to_datetime(df.date).dt.strftime('%Y%m%d')

#%%
lookback_window     = 48
rebal_period        = 3
start_date          = '19770131'
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
saa_return  = np.array([])
bl_return   = np.array([])


for date in tdates:
    # Create subset of data where there are at least as many values as required by the lookback window
    # I.e. only consider assets that have at enough past returns for a given date    
    if date in rebal_dates:
        # Get necessary data
        universe        = df.loc[(df.date <= date)][-lookback_window:].dropna(axis='columns')      
        returns         = universe.drop(columns=['date', 'Cash CHF'])
        rf              = universe['Cash CHF'].drop(columns=['date']).values[-2]
        returns_t       = returns.values[-1]
        returns_p       = returns[:-1]
        assets          = returns.columns
        weights         = (df_w.loc[:, assets].values[0])
        
        # Mean-Variance Optimization        
        myopt           = optimizers2.MeanVariance(rf=rf, permnos=assets, returns=returns_p.T, rebal_period=rebal_period, mean_pred='Holt')
        # _,_,W_m,_,_     = myopt.efficient_frontier()

        # 1/N
        W_ev            = np.ones([len(assets)]) / len(assets)
        
        # Black-Litterman
        W_saa           = weights/(sum(weights))
        mybl            = optimizers2.BlackLitterman(rf=0,permnos=assets, returns=returns_p.T, rebal_period=rebal_period, market_weights=W_saa)
        P               = np.eye(len(assets))
        q               = np.expand_dims(myopt.R, axis=1)
        O               = np.diag(myopt.C.diagonal())
        tau             = 0.01
        mybl.get_model_return(tau=tau, P=P, O=O, q=q)
        _,_,W_bl,_,_     = mybl.efficient_frontier_bl()

    # 
    universe        = df.loc[(df.date == date)]
    returns_t       = np.squeeze(universe[universe.columns[universe.columns.isin(assets)]].values)

    W_saa           = (W_saa * (1+returns_t)) / np.dot(W_saa, 1+returns_t)
    # W_m             = (W_m * (1+returns_t)) / np.dot(W_m, 1+returns_t)
    W_ev            = (W_ev * (1+returns_t)) / np.dot(W_ev, 1+returns_t)
    W_bl            = (W_bl * (1+returns_t)) / np.dot(W_bl, 1+returns_t)

    # m_return        = np.append(m_return, W_m.dot(returns_t))
    ev_return       = np.append(ev_return, W_ev.dot(returns_t))
    saa_return      = np.append(saa_return, W_saa.dot(returns_t))
    bl_return       = np.append(bl_return, W_bl.dot(returns_t))

#%%
# m_value     = 1+m_return
ev_value    = 1+ev_return
saa_value   = 1+saa_return
bl_value    = 1+bl_return

# m_value     = np.cumprod(m_value, axis=0)
ev_value    = np.cumprod(ev_value, axis=0)
saa_value   = np.cumprod(saa_value, axis=0)
bl_value    = np.cumprod(bl_value, axis=0)

plt.plot(ev_value, label='1/N', color='blue')
# plt.plot(m_value, label='MV', color='green')
plt.plot(saa_value, label='SAA', color='red')
plt.plot(bl_value, label='TAA(BL)', color='white')

plt.legend()
plt.yscale('log')




# %%
height=W_bl-W_saa
bars = assets
y_pos = np.arange(len(bars))
plt.bar(y_pos, height)
plt.xticks(y_pos, bars, rotation='vertical')


# %%
