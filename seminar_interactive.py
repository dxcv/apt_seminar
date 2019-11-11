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
lookback_window     = 49
rebal_period        = 3
start_date          = '19780131'
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
W_m_W_saa   = []
W_gmv_W_saa = []


# Define parameters for the BL models
MeanModel1      = 'MLE'
MeanMOdel2      = 'Holt'
tau             = 0.02
uncertainty     = 0.2

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
        myopt           = optimizers2.MeanVariance(rf=0, permnos=assets, returns=returns_p, rebal_period=rebal_period, var_pred='MLE')
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
        mybl2.get_model_return(tau=tau, P=P, O=O, q=q)
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

    # Collect weight differences to SAA
    if date in wdates:
        W_bl1_W_saa.append(W_bl1-W_saa)
        W_bl2_W_saa.append(W_bl2-W_saa)
        W_m_W_saa.append(W_m-W_saa)
        W_gmv_W_saa.append(W_gmv-W_saa)

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

tdates2   = [datetime.datetime.strptime(str(x)[0:8], '%Y%m%d') for x in tdates]
plt.plot(tdates2, m_value, label='MV', color='green')
plt.plot(tdates2, ev_value, label='1/N', color='blue')
plt.plot(tdates2, gmv_value, label='gMinVar', color='orange')
plt.plot(tdates2, saa_value, label='SAA', color='black')
plt.plot(tdates2, bl1_value, label='TAA(BL '+MeanModel1+')', color='red')
plt.plot(tdates2, bl2_value, label='TAA(BL '+MeanMOdel2+')', color='magenta')
plt.title('Performance Chart with '+r'$\tau$='+str(tau).replace('\t', ' ')+' and $\omega$='+str(uncertainty))
plt.legend()
plt.yscale('log')
plt.show()

#%% Save returs to a excel file
labels = ['MaxSR-(0,1)', '1/N', 'gMinVar', 'SAA', 'TAA(BL '+MeanModel1+')', 'TAA(BL '+MeanMOdel2+')']

model_returns = pd.DataFrame(
    {labels[0]: m_return,
     labels[1]: ev_return,
     labels[2]: gmv_return,
     labels[3]: saa_return,
     labels[4]: bl1_return,
     labels[5]: bl2_return})

model_returns.to_excel('Model returns with tau ='+str(tau)+' + omega='+str(uncertainty)+'.xlsx')


#%% Plot model weights deviations from SAA

#%%
w_dates   = [datetime.datetime.strptime(str(x)[0:8], '%Y%m%d') for x in wdates]

fig, axs = plt.subplots(2, 2)
# BL1
axs[0, 0].plot(w_dates, W_bl1_W_saa)
axs[0, 0].set_title(labels[4])

# BL2
axs[0, 1].plot(w_dates, W_bl2_W_saa)
axs[0, 1].set_title(labels[5])

# cMV
axs[1, 0].plot(w_dates, W_m_W_saa)
axs[1, 0].set_title(labels[0])

# gMV
axs[1, 1].plot(w_dates, W_gmv_W_saa)
axs[1, 1].set_title(labels[2])

# for ax in axs.flat:
#     ax.set(xlabel='x-label', ylabel='y-label')

# Hide x labels and tick labels for top plots and y ticks for right plots.
# for ax in axs.flat:
#     ax.label_outer()

plt.show()

#%% Summary risk statistics
from scipy.stats import kurtosis, skew
all_returns     = [m_return, gmv_return, ev_return, saa_return, bl1_return, bl2_return]
all_values      = [m_value, gmv_value, ev_value, saa_value, bl1_value, bl2_value]
geo_returns     = []
std_devs        = []
sharpe_ratios   = []
kurtos          = []
skews           = []
valueatrisk     = []
exp_shortfall   = []


for i in all_values:

    #%% Geometric Return
    geo_i = round((1+(i[-1]/i[0])**(1/len(i))-1)**12-1, ndigits=4)
    geo_returns.append(geo_i)

for i in all_returns: 
    
    #%% Standard Deviation
    std_i = round((np.std(i)*np.sqrt(12)), ndigits=4)
    std_devs.append(std_i)

    #%% Value at Risk
    var_i = round(np.quantile(i, 0.01), ndigits=4)
    valueatrisk.append(var_i)

    #%% Expected Shortfall
    vars = []
    for alpha in np.linspace(0.00001, 0.01):
        vars.append(round(np.quantile(i, alpha), ndigits=4))
    es_i = round(np.mean(vars), ndigits=4)
    exp_shortfall.append(es_i)

    # %% Sharpe Ratio
    def ann_sr(rets, rfrets):
        return ((1+np.mean(rets-rfrets))**12-1)/(np.std(rets)*np.sqrt(12))

    sr_i = round(ann_sr(i, rf_return), ndigits=4)
    sharpe_ratios.append(sr_i)

    #%% Kurtosis
    kurtos.append(round(skew(i), ndigits=4))

    #%% Skewness
    skews.append(round(kurtosis(i), ndigits=4))


#%%
labels = ['MaxSR-(0,1)', 'gMinVar', '1/N',  'SAA', 'TAA(BL '+MeanModel1+')', 'TAA(BL '+MeanMOdel2+')']

summary_stats = pd.DataFrame(
    {'gMean p.a.': geo_returns,
     'StdDev p.a.': std_devs,
     'SR p.a.': sharpe_ratios,
     'Kurtosis': kurtos,
     'Skewness': skews,
     '1%-VaR': valueatrisk,
     r'1%-ES': exp_shortfall}, index=labels)

summary_stats.to_excel('Summary stats wit tau='+str(tau)+'+omega='+str(uncertainty)+'.xlsx')
summary_stats

# # %%
# height=W_m - W_ev
# bars = assets
# y_pos = np.arange(len(bars))
# plt.bar(y_pos, height)
# plt.xticks(y_pos, bars, rotation='vertical')



