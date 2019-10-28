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
importlib.reload(datapreprocessing2)
path    = 'C:/Users/silva/iCloudDrive/Docs/Ausbildung/QuantLibrary/MScQF_Thesis/9. APT Seminar/Returns_monthly2.csv'
data    = datapreprocessing2.Preprocessing(path)

#%% Compute for each possible point in time within the data the n largest stocks based on market cap
data.compute_nlargest(10)

#%%
lookback_window     = 48
rebal_period        = 3

permnos, returns, caps, market_weights = data.permnos_returns_caps_weights(date=20181231, lookback_window=lookback_window)

#%%

importlib.reload(optimizers2)
myopt = optimizers2.MeanVariance(rf=0, permnos=permnos, returns=returns, rebal_period=rebal_period)
mybl = optimizers2.BlackLitterman(rf=0,permnos=permnos, returns=returns, rebal_period=rebal_period, market_weights=market_weights)
# myopt.display_frontier()
# plt.plot(front_var, front_ret)


#%%
#========================================
# Efficient Frontier and Tangency Portfolio
#========================================
importlib.reload(optimizers2)

myopt.display_frontier(label='Historical returns')
myopt.display_assets()

mybl.display_frontier(label='Implied returns', color='red')
mybl.display_assets(color='red')

plt.xlabel('Risk $\sigma$'), plt.ylabel('Return $\mu$'), plt.legend()


#%%
#==================================
# Mean-variance portfolio
#==================================
                   
delta = 3               
                   
w                   = np.linalg.solve(delta * myopt.C, myopt.R)
w_m                 = market_weights
_, _, w_tan, _, _   = myopt.efficient_frontier()
n = len(market_weights)


fig, ax = plt.subplots(figsize = (8, 5))
ax.set_title('Overview portfolio weights with $\delta$ = '+str(delta), fontsize = 12)
ax.plot(np.arange(n) + 1, w, 'o', color = 'b', label = '$w$ (MV)')
ax.plot(np.arange(n) + 1, w_m, 'o', color = 'r', label = '$w_m$ (Market weights)')
ax.plot(np.arange(n) + 1, w_tan, 'o', color = 'g', label = '$w_t$ (Tangency weights)')
ax.vlines(np.arange(n) + 1, 0, w, lw = 1)
ax.vlines(np.arange(n) + 1, 0, w_m, lw = 1)
#ax.vlines(np.arange(n) + 1, 0, w_t, lw = 1)
ax.axhline(0, color = 'k')
ax.axhline(-1, color = 'k', linestyle = '--')
ax.axhline(1, color = 'k', linestyle = '--')
ax.set_xlim([0, n+1])
ax.set_xlabel('Assets', fontsize = 11)
ax.xaxis.set_ticks(np.arange(1, n + 1, 1))
plt.legend(numpoints = 1, loc = 'best', fontsize = 11)
plt.show()
# pd.DataFrame({'Tangency Weight': w_tan}, index=myopt.permnos).T
# pd.DataFrame({'Mean Variance Weight': w}, index=myopt.permnos).T



#%%
# Calculate portfolio historical return and variance


fig, ax = plt.subplots(figsize = (8, 5))
ax.set_title(r'Difference between $\hat{\mu}$ (historical) and $\mu_{BL}$ (market implied)', fontsize = 12)
ax.plot(np.arange(n) + 1, myopt.R, 'o', color = 'b', label = '$\hat{\mu}$')
ax.plot(np.arange(n) + 1, mybl.R, 'o', color = 'r', label = '$\mu_{BL}$')
ax.vlines(np.arange(n) + 1, mybl.R, myopt.R, lw = 1, color='white')
ax.axhline(0, color = 'white', linestyle = '--')
ax.set_xlim([0, n + 1])
ax.set_xlabel('Asset')
plt.legend(numpoints = 1, loc = 'best', fontsize = 11)
plt.show()

# pandas.DataFrame({'Impled Returns (BL)': impl_R,'Historical Returns': R}, index=names).T

#%%
# With Views P,q and confidence omega and tau
from numpy import matrix, array, zeros, empty, sqrt, ones, dot, append, mean, cov, transpose, linspace, eye
from numpy.linalg import inv, pinv
P = [[1,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,1,0,0,0,0],
    [0,0,-1,0,0,1,0,0,0,0],
    [0,0,0,0,0,1,0,-1,0,0],
    [1,0,0,0,-0.2,0,0.2,0,-1,0]]
q = [[-0.03],
     [0.04],
     [0.02],
     [0.01],
     [0.01]]
omega = [0.1,0.3,0.5,0.7,0.9]*ones(5)
OMEGA = omega*eye(5)
tau = 0.3
mu_combined2    = dot(inv(inv(tau*mybl.C)+dot(dot(np.transpose(P),inv(OMEGA)),P)),(dot(inv(tau*mybl.C),np.transpose([mybl.R]))+dot(dot(np.transpose(P),inv(OMEGA)),q)))
w_combined2     = dot(inv(3*mybl.C),mu_combined2)

part1           = myopt.R
part2           = np.dot(tau*myopt.C, np.transpose(P))
part3           = inv(np.dot(np.dot(np.dot(tau,P), myopt.C), np.transpose(P)) + OMEGA)
part4           = (np.asarray(q).flatten()-np.dot(P, myopt.R))
mu_combined     = part1+np.dot(np.dot(part2, part3),part4)

w_combined      = dot(1/3*inv(mybl.C),mu_combined2)

# mu_combined2    = mu_combined2.flatten()
# w_combined2     = w_combined2.flatten()

# pd.DataFrame({'BL Returns': mu_combined2,'BL Weights': w_combined2}, index=permnos).T

#%%
print(np.sum(w_combined))

#%%


#%%
