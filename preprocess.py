#%%
import pandas as pd

#%%
path_sp = 'C:/Users/silva/iCloudDrive/Docs/Ausbildung/QuantLibrary/MScQF_Thesis/9. APT Seminar/SP_monthly.csv'
path_rf = 'C:/Users/silva/iCloudDrive/Docs/Ausbildung/QuantLibrary/MScQF_Thesis/9. APT Seminar/RF_monthly.csv'


#%%
df = pd.read_csv('Returns_short.csv')
df['CAP'] = df.PRC*df.SHROUT
df['RET+1'] = df['RET'].shift(-1)
df = df.loc[(df.RET!='C') & (df.RET!='B') & (df['RET+1']!='C') & (df['RET+1']!='B')]
df = df.dropna(subset=['RET+1', 'CAP'])
df = df.drop(columns=['RCRDDT', 'SHROUT', 'PERMCO'])
df[['RET', 'RET+1']].astype(float)
df.to_csv('Returns_final.csv',index=False)

#%%




#%%
sp = pd.read_csv(path_sp)
rf = pd.read_csv(path_rf)

#%%
sp = sp[['caldt', 'sprtrn']].dropna()
rf = rf[['MCALDT', 'TMYTM']].dropna()

#%%
rf = rf.rename(columns={"MCALDT": "date", 'TMYTM': 'rfrtrn'})
sp = sp.rename(columns={"caldt": "date"})

#%%
rf.rfrtrn = ((rf.rfrtrn/100)+1)**(1/12)-1

#%%
rf.date= pd.to_datetime(rf.date)
rf.date = rf.date.dt.strftime('%Y%m%d')

#%%
rf.date = rf.date.astype(int)
sp.date = sp.date.astype(int)

#%%
rf.head()

#%%

sp.set_index('date', inplace=True)
rf.set_index('date', inplace=True)

#%%

