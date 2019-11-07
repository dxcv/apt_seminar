#%% Load packages
import numpy as np
import pandas as pd
import datetime
import logging
import datapreprocessing2 as dp
import optimizers2
import matplotlib.pyplot as plt
import importlib

#%% Load data
df          = pd.read_excel('Seminar_returns.xlsx', sheet_name='time series')
df_w        = pd.read_excel('Seminar_returns.xlsx', sheet_name='market caps')
df.date     = pd.to_datetime(df.date).dt.strftime('%Y%m%d')

#%%
lookback_window     = 48
rebal_period        = 3
start_date          = '20140131'
end_date            = '20190831'
int_date            = '20140131'

dates               = df.date.unique()
start_position      = np.where(dates == start_date)[0].item(0)
end_position        = np.where(dates == end_date)[0].item(0)
obs_start           = dates[start_position-lookback_window+1]
tdates              = df.loc[(df.date >= start_date) & (df.date <= end_date),'date'].values
wdates              = df.loc[(df.date >= int_date) & (df.date <= end_date),'date'].values
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
# Initialize return arrays
importlib.reload(optimizers2)
m_return    = np.array([])
ev_return   = np.array([])
saa_return  = np.array([])
gmv_return  = np.array([])
rf_return   = np.array([])
bl1_return  = np.array([])
bl2_return  = np.array([])

# Initialize weight differences
W_bl1_W_saa = []
W_bl2_W_saa = []

# Define parameters for the BL models
MeanModel1      = 'MLE'
MeanMOdel2      = 'Holt'
tau             = 0.02
uncertainty     = 0.5

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
        myopt           = optimizers2.MeanVariance(rf=0, permnos=assets, returns=returns_p, rebal_period=rebal_period)
        _,_,W_m,_,_     = myopt.efficient_frontier()

        # Global minimum variance portfolio
        W_gmv           = myopt.min_variance()

        # 1/N
        W_ev            = np.ones([len(assets)]) / len(assets)
        
        # Black-Litterman MLE
        W_saa           = weights/(sum(weights))
        mybl1           = optimizers2.BlackLitterman(rf=0, permnos=assets, returns=returns_p, rebal_period=rebal_period, market_weights=W_saa)
        myview1         = optimizers2.MeanVariance(rf=0, permnos=assets, returns=returns_p, rebal_period=rebal_period, mean_pred=MeanModel1)
        P               = np.eye(len(assets))
        q               = np.expand_dims(myview1.R, axis=1)
        O               = np.sqrt(np.diag(myview1.C.diagonal()))
        mybl1.get_model_return(tau=tau, P=P, O=O, q=q)
        _,_,W_bl1,_,_   = mybl1.efficient_frontier_bl()

        # Black-Litterman HOLT
        W_saa           = weights/(sum(weights))
        mybl2           = optimizers2.BlackLitterman(rf=0, permnos=assets, returns=returns_p, rebal_period=rebal_period, market_weights=W_saa)
        myview2         = optimizers2.MeanVariance(rf=0, permnos=assets, returns=returns_p, rebal_period=rebal_period, mean_pred=MeanMOdel2)
        P               = np.eye(len(assets))
        q               = np.expand_dims(np.ones(len(assets)), axis=1)*uncertainty
        O               = np.sqrt(np.diag(myview2.C.diagonal()))
        # mybl2.get_model_return(tau=tau, P=P, O=O, q=q)
        _,_,W_bl2,_,_   = mybl2.efficient_frontier_bl()

    # Update returns and portfolio weights 
    universe        = df.loc[(df.date == date)]
    returns_t       = np.squeeze(universe[universe.columns[universe.columns.isin(assets)]].values)
    
    W_m             = (W_m * (1+returns_t)) / np.dot(W_m, 1+returns_t)
    W_gmv           = (W_gmv * (1+returns_t)) / np.dot(W_gmv, 1+returns_t)
    W_ev            = (W_ev * (1+returns_t)) / np.dot(W_ev, 1+returns_t)
    W_saa           = (W_saa * (1+returns_t)) / np.dot(W_saa, 1+returns_t)
    W_bl1           = (W_bl1 * (1+returns_t)) / np.dot(W_bl1, 1+returns_t)
    W_bl2           = (W_bl2 * (1+returns_t)) / np.dot(W_bl2, 1+returns_t)

    m_return        = np.append(m_return, W_m.dot(returns_t))
    gmv_return      = np.append(gmv_return, W_gmv.dot(returns_t))  
    ev_return       = np.append(ev_return, W_ev.dot(returns_t))
    saa_return      = np.append(saa_return, W_saa.dot(returns_t))
    bl1_return      = np.append(bl1_return, W_bl1.dot(returns_t))  
    bl2_return      = np.append(bl2_return, W_bl2.dot(returns_t))  

    rf_return       = np.append(rf_return, rf)

    # Collect weight differences from TAA to SAA
    if date in wdates:
        W_bl1_W_saa.append(W_bl1-W_saa)
        W_bl2_W_saa.append(W_bl2-W_saa)


#%%
m_value     = 1+m_return
gmv_value   = 1+gmv_return
ev_value    = 1+ev_return
saa_value   = 1+saa_return
bl1_value   = 1+bl1_return
bl2_value   = 1+bl2_return

m_value     = np.cumprod(m_value, axis=0)
gmv_value   = np.cumprod(gmv_value, axis=0)
ev_value    = np.cumprod(ev_value, axis=0)
saa_value   = np.cumprod(saa_value, axis=0)
bl1_value   = np.cumprod(bl1_value, axis=0)
bl2_value   = np.cumprod(bl2_value, axis=0)


# labels = ['MV', '1/N', 'gMinVar', 'SAA', 'TAA(BL '+Model1+')', ]

tdates2   = [datetime.datetime.strptime(str(x)[0:8], '%Y%m%d') for x in tdates]
plt.plot(tdates2, m_value, label='MV', color='green')
plt.plot(tdates2, ev_value, label='1/N', color='blue')
plt.plot(tdates2, gmv_value, label='gMinVar', color='orange')
plt.plot(tdates2, saa_value, label='SAA', color='red')
plt.plot(tdates2, bl1_value, label='TAA(BL '+MeanModel1+')', color='white')
plt.plot(tdates2, bl2_value, label='TAA(BL '+MeanMOdel2+')', color='magenta')
plt.title('Performance Chart with '+r'$\tau$='+str(tau).replace('\t', ' ')+' and $\omega$='+str(uncertainty))
plt.legend()
plt.yscale('log')


#%%
test1 = np.array(W_bl1_W_saa)

#%%

plt.plot(test1)

#%%
all_returns = [m_return, gmv_return, ev_return, saa_return, bl1_return, bl2_return]







#%% Geometric Return
all_values = [m_value, gmv_value, ev_value, saa_value, bl1_value, bl2_value]
for i in all_values:
    geometric_ret = (1+(i[-1]/i[0])**(1/len(i))-1)**12-1
    print('gMean p.a.:'+str(geometric_ret))
 
#%% Standard Deviation
for i in all_returns:
    print('StDev p.a.: '+str(np.std(i)*np.sqrt(12)))

#%% Value at Risk
for i in all_returns:
    print('1%-VaR: '+str(round(np.quantile(i, 0.01), ndigits=4)))

# %% Expected Shortfall

for i in all_returns:
    vars = []
    for alpha in np.linspace(0.00001, 0.01):
        vars.append(round(np.quantile(i, alpha), ndigits=4))
    es = np.mean(vars)
    print('1%-'+'ES: '+str(np.round(es, 4)))

# %% Sharpe Ratio
def ann_sr(rets, rfrets):
    return ((1+np.mean(rets-rfrets))**12-1)/(np.std(rets)*np.sqrt(12))

for i in all_returns:
    print('SR p.a.: '+str(ann_sr(i, rf_return)))

#%% Kurtosis
from scipy.stats import kurtosis, skew
for i in all_returns:
    print('Skewness: '+str(skew(i)))

for i in all_returns:
    print('Kurtosis: '+str(kurtosis(i)))

#%%

# %%
height=W_bl2-W_saa
bars = assets
y_pos = np.arange(len(bars))
plt.bar(y_pos, height)
plt.xticks(y_pos, bars, rotation='vertical')



# %%
