#%%
import numpy as np
import pandas as pd
import datetime



#%%
df = pd.read_excel('Seminar_returns.xlsx')
df.Datum = pd.to_datetime(df.Datum).dt.strftime('%Y%m%d')

df.Datum


#%%
lookback_window     = 10
rebal_period        = 3
start_date          = '20190131'
end_date            = '20190831'
dates               = df.Datum.unique()
start_position      = np.where(dates == start_date)[0].item(0)
end_position        = np.where(dates == end_date)[0].item(0)
tdates              = dates[start_position:end_position+1]
odates              = dates[start_position-lookback_window+1:start_position+1]
# tdates              = df.loc[(df.Datum >= start_date) & (df.Datum <= end_date),'Datum'].values
horizon             = len(tdates)
max_rebal_period    = len(dates[start_position:end_position])


if rebal_period == 1:
    mod = divmod(horizon, rebal_period)[0]
elif rebal_period > max_rebal_period:
    rebal_period = max_rebal_period
    mod = divmod(horizon, rebal_period)[0]
else:
    mod = divmod(horizon, rebal_period)[0]+1
    

print(tdates)
#%%
counter     = 0
rebal_dates = []

for i in range(mod):
    rebal_date = tdates[counter]
    rebal_dates.append(rebal_date)
    counter += rebal_period
print(rebal_dates)

# #%%
# for date in tdates:
#     if date in rebal_dates:
#         print(df.loc[(df.Datum <= date), 'Gold'].values)
#     else:
#         pass





#%% 
df2 = df.copy().drop(columns=['Liability Proxy 1-3 years', 'Liability Proxy 3-5 years', 'Liability Proxy 5-7 years', 'Liability Proxy 10+ years'])
