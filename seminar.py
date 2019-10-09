#%%
import numpy as np
import pandas as pd
import datetime



#%%
df = pd.read_excel('Seminar_returns.xlsx')


#%%
lookback_window     = 3
rebal_period        = 12
start_date          = '31.01.2018'
end_date            = '31.08.2010'
dates               = df.Datum.unique()
# dates               = datetime.datetime.strptime(df.Datum.unique().item(), '%d%m%Y')
# start_position      = np.where(dates == start_date)[0].item(0)
# end_position        = np.where(dates == end_date)[0].item(0)
# tdates              = dates[start_position:end_position]
# tdates               = dates[(dates<=end_date)]
horizon             = len(tdates)
min_rebal_period    = horizon % lookback_window + 1


print(type(dates))

#%%
counter = lookback_window
rebal_dates = []
mod = 


for i in range(10):
    rebal_date = dates[counter]
    rebal_dates.append(rebal_date)
    counter += lookback_window

print(rebal_dates[-1:])





#%% 
df2 = df.copy().drop(columns=['Liability Proxy 1-3 years', 'Liability Proxy 3-5 years', 'Liability Proxy 5-7 years', 'Liability Proxy 10+ years'])
