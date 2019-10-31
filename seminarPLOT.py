
import numpy as np
import pandas as pd
import datetime
import logging
import datapreprocessing2
import optimizers2
import matplotlib.pyplot as plt
import importlib


importlib.reload(datapreprocessing2)
path    = 'C:/Users/silva/iCloudDrive/Docs/Ausbildung/QuantLibrary/MScQF_Thesis/9. APT Seminar/Returns_monthly2.csv'
data    = datapreprocessing2.Preprocessing(path)

data.compute_nlargest(10)
data.save_nlargest()
lookback_window     = 48
rebal_period        = 12

permnos, returns, caps, market_weights = data.permnos_returns_caps_weights(date=20181231, lookback_window=lookback_window)

# importlib.reload(optimizers2)
# myopt = optimizers2.MeanVariance(rf=0, permnos=permnos, returns=returns, rebal_period=rebal_period)
# mybl1 = optimizers2.BlackLitterman(rf=0,permnos=permnos, returns=returns, rebal_period=rebal_period, market_weights=market_weights)
# mybl2 = optimizers2.BlackLitterman(rf=0,permnos=permnos, returns=returns, rebal_period=rebal_period, market_weights=market_weights)

# #Plot 1
# delta = 0.8 
# myopt.display_frontier(label='Historical returns')
# myopt.display_assets()
# mybl.display_frontier(label='Implied returns', color='red')
# mybl.display_assets(color='red')
# plt.title('Efficient frontiers as of 31.12.2018 with $\lambda=$'+str(delta))
# plt.xlabel('Risk $\sigma$')
# plt.ylabel('Return $\mu$')
# plt.legend()
# plt.show()

# #Plot 2
# delta = 0.8               
                   
# w                   = np.linalg.solve(delta * myopt.C, myopt.R)
# w_m                 = market_weights
# _, _, w_tan, _, _   = myopt.efficient_frontier()
# n = len(market_weights)


# fig, ax = plt.subplots(figsize = (8, 5))
# ax.set_title('Overview portfolio weights with $\lambda$ = '+str(delta), fontsize = 12)
# ax.plot(np.arange(n) + 1, w, 'o', color = 'b', label = '$w$ (MV)')
# ax.plot(np.arange(n) + 1, w_m, 'o', color = 'r', label = '$w_m$ (Market weights)')
# ax.plot(np.arange(n) + 1, w_tan, 'o', color = 'g', label = '$w_t$ (Const. tangency weights)')
# ax.vlines(np.arange(n) + 1, 0, w, lw = 1)
# ax.vlines(np.arange(n) + 1, 0, w_m, lw = 1)
# #ax.vlines(np.arange(n) + 1, 0, w_t, lw = 1)
# ax.axhline(0, color = 'k')
# ax.axhline(-1, color = 'k', linestyle = '--')
# ax.axhline(1, color = 'k', linestyle = '--')
# ax.set_xlim([0, n+1])
# ax.set_xlabel('Assets', fontsize = 11)
# ax.xaxis.set_ticks(np.arange(1, n + 1, 1))
# plt.legend(numpoints = 1, loc = 'best', fontsize = 11)
# plt.show()

#Plot 3
importlib.reload(optimizers2)
myopt = optimizers2.MeanVariance(rf=0, permnos=permnos, returns=returns, rebal_period=rebal_period)
mybl1 = optimizers2.BlackLitterman(rf=0,permnos=permnos, returns=returns, rebal_period=rebal_period, market_weights=market_weights)
mybl2 = optimizers2.BlackLitterman(rf=0,permnos=permnos, returns=returns, rebal_period=rebal_period, market_weights=market_weights)

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

mybl2.get_model_return(tau=tau, P=P, O=OMEGA, q=q)

mybl1.display_frontier_bl(label='Implied returns', color='red')
mybl2.display_frontier_bl(label='Combined returns', color='green')
mybl1.display_assets_bl(color='red')
mybl2.display_assets_bl(color='green')
plt.xlabel('Risk $\sigma$')
plt.ylabel('Return $\mu$')
plt.legend(loc='upper left')
plt.title(r'Influence of views and $\tau$='+str(tau).replace('\t', ' '))
plt.show()

#Plot 4
