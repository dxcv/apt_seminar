import numpy as np
import pandas as pd
import json
from sklearn.covariance import LedoitWolf

class Preprocessing:
    
    # Initializer / Instance Attributes
    def __init__(self, path):
        if '.csv' in path:
            self.df        = pd.read_csv(path)
        else:
            self.df        = pd.read_excel(path)
            self.df.date   = pd.to_datetime(df.date).dt.strftime('%Y%m%d')

        self.nlargest      = {}
        self.trading_dates = np.sort(self.df.date.unique())
        
    def compute_nlargest(self, n):
        # Compute n largest stocks, based on market cap, for each day in the data
        self.nlargest   = self.df.groupby([pd.Grouper(key='date'), 'PERMNO'])['CAP'].sum()
        self.nlargest   = self.nlargest.groupby(level=0).nlargest(n).sort_index().reset_index(level=1, drop=True).to_frame()
        self.nlargest   = self.nlargest.groupby(level=0).apply(lambda nlargest: nlargest.xs(nlargest.name).to_dict()).to_dict()
        
    def save_nlargest(self):
        # Save dict with n largest stocks per date to json file
        with open('nlargest_caps.json', 'w') as fp:
            json.dump(self.nlargest, fp)
            
    def get_trading_interval(self, date, lookback_window):
        return self.trading_dates[self.trading_dates <= date][-lookback_window:]
            
    def get_reduced_data(self, date, lookback_window):
        # Get list that includes a dict with the largest n caps of that day
        nlargeststocks = [*self.nlargest[date].values()]
        nlargeststocks = [*nlargeststocks[0].keys()]
        trading_interval = self.get_trading_interval(date, lookback_window)
        # Return reduced data file (includies only relevant stocks for the relevant interval)
        return self.df[(self.df.PERMNO.isin(nlargeststocks)) & (self.df.date.isin(trading_interval))].reset_index(drop=True)

    
    
    def permnos_returns_caps_weights(self, date, lookback_window):
        raw = self.get_reduced_data(date=date, lookback_window=lookback_window)[['date','PERMNO', 'RET', 'RET+1']]
        permnos = raw.PERMNO.unique()
        names = self.nlargest[date]['CAP']
        returns, caps, returns_ahead, deleted = [], [], [], []
        for name in permnos:
            if len(np.array(raw.loc[raw.PERMNO == name, 'RET'].values)) == lookback_window:
                returns.append(np.array(raw.loc[raw.PERMNO == name, 'RET'].values))
                caps.append(names[name]) 
                returns_ahead.append(np.array(raw.loc[(raw.PERMNO==name) & (raw.date==date), 'RET+1'].values))
                market_weights = np.array(caps) / sum(caps)
            else:
                deleted.append(name)
        permnos = np.delete(permnos, deleted)
        return permnos, returns, caps, market_weights, returns_ahead
    
    def means_historical(self, date, lookback_window):
        permnos, returns, _, market_weights, returns_ahead = self.permnos_returns_caps_weights(date, lookback_window)
        returns = np.matrix(returns)
        rows, _ = returns.shape
        exp_returns = np.array([])
        for r in range(rows):
            exp_returns = np.append(exp_returns, np.mean(returns[r]))
        covars = LedoitWolf().fit(np.transpose(returns)).covariance_
        #covars = np.cov(returns)
        #correls = np.corrcoef(returns)
        #self.exp_returns = (1 + self.exp_returns) ** 250 - 1  # 3months returns
        #self.covars = self.covars * 250  # 3months covariances
        return permnos, market_weights, exp_returns, covars, returns_ahead
    
    def return_summary(self, date, lookback_window):
        permnos, market_weights, exp_returns, covars, _ = self.means_historical(date, lookback_window)
        mean_weight_overview = pd.DataFrame({'Return': exp_returns, 'Market Weight': market_weights}, index=permnos).T
        cov_overview = pd.DataFrame(covars, columns=permnos, index=permnos)
        #cor_overview = pd.DataFrame(correls, columns=permnos, index=permnos)
        return mean_weight_overview, cov_overview