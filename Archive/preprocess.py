#%%
import pandas as pd

#%%
path_sp = 'C:/Users/silva/iCloudDrive/Docs/Ausbildung/QuantLibrary/MScQF_Thesis/9. APT Seminar/SP_monthly.csv'
path_rf = 'C:/Users/silva/iCloudDrive/Docs/Ausbildung/QuantLibrary/MScQF_Thesis/9. APT Seminar/RF_monthly.csv'


#%%
path = 'C:/Users/silva/iCloudDrive/Docs/Ausbildung/QuantLibrary/MScQF_Thesis/9. APT Seminar/Returns_monthly.csv'
target = 'C:/Users/silva/iCloudDrive/Docs/Ausbildung/QuantLibrary/MScQF_Thesis/9. APT Seminar/Returns_monthly2.csv'
df = pd.read_csv(path)

#%%
from datetime import datetime
# date = datetime.strptime(str()[0:8], '%Y%m%d')
df.date = [datetime.strptime(str(x)[0:8], '%Y%m%d') for x in df.date]
df.date   = pd.to_datetime(df.date).dt.strftime('%Y%m%d')
df['CAP'] = df.PRC*df.SHROUT
# df['RET+1'] = df['RET'].shift(-1)
# df = df.loc[(df.RET!='C') & (df.RET!='B') & (df['RET+1']!='C') & (df['RET+1']!='B')]
df = df.loc[(df.RET!='C') & (df.RET!='B')]
df = df.dropna(subset=['CAP'])
df['RET'].astype(float)
df.to_csv(target,index=False, date_format='%Y%m%d')

#%%
df2 = pd.read_csv(path, parse_dates=True)

#%%
df2.date

#%%
# sp = pd.read_csv(path_sp)
# rf = pd.read_csv(path_rf)

# #%%
# sp = sp[['caldt', 'sprtrn']].dropna()
# rf = rf[['MCALDT', 'TMYTM']].dropna()

# #%%
# rf = rf.rename(columns={"MCALDT": "date", 'TMYTM': 'rfrtrn'})
# sp = sp.rename(columns={"caldt": "date"})

# #%%
# rf.rfrtrn = ((rf.rfrtrn/100)+1)**(1/12)-1

# #%%
# rf.date= pd.to_datetime(rf.date)
# rf.date = rf.date.dt.strftime('%Y%m%d')

# #%%
# rf.date = rf.date.astype(int)
# sp.date = sp.date.astype(int)

# #%%
# rf.head()

# #%%

# sp.set_index('date', inplace=True)
# rf.set_index('date', inplace=True)

# #%%

