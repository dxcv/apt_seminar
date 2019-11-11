#%%
import numpy as np
import pandas as pd
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy import stats

#%%
df = pd.read_excel('Seminar_returns.xlsx')

#%% 
df2 = df.copy().drop(columns=['Liability Proxy 1-3 years', 'Liability Proxy 3-5 years', 'Liability Proxy 5-7 years', 'Liability Proxy 10+ years'])

#%%
df2.head()


#%%
summary = []

returns     = df2.dropna().drop(columns=['Datum'])

mean        = ((1+np.mean(returns))**12-1)*100
std         = (np.std(returns)*(12**0.5))*100
skewness    = skew(returns)
kurt        = kurtosis(returns)
jb_ts       = stats.jarque_bera(returns)[0]
jb_pv       = stats.jarque_bera(returns)[1]
intermed    = [mean, std, skewness, kurt, jb_ts, jb_pv]

print(skewness)




#%%
