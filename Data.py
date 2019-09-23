import numpy as np
import pandas as pd
import json

class Data:
    
    # Initializer / Instance Attributes
    def __init__(self, path):
        self.df            = pd.read_csv(path)
        self.df['CAP']     = self.df.PRC*self.df.SHROUT
        self.df['RET+1']   = self.df['RET'].shift(-1)
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
        self.nlargeststocks = [*self.nlargest[date].values()]
        # Get list with keys(PERMNOs)
        self.nlargeststocks = [*self.nlargeststocks[0].keys()]
        # Get trading interval
        self.trading_interval = self.get_trading_interval(date, lookback_window)
        # Return reduced data file (includies only relevant stocks for the relevant interval)
        return self.df[(self.df.PERMNO.isin(self.nlargeststocks)) & (self.df.date.isin(self.trading_interval))].reset_index(drop=True)
    
    def permnos_returns_caps_weights(self, date, lookback_window):
        self.raw = self.get_reduced_data(date=date, lookback_window=lookback_window)[['PERMNO', 'RET']]
        self.permnos = self.raw.PERMNO.unique()
        self.names = self.nlargest[date]['CAP']
        self.returns, self.caps = [], []
        for name in self.permnos:
            self.returns.append(self.raw.loc[self.raw.PERMNO == name, 'RET'].astype(float))
            self.caps.append(self.names[name])  
        self.market_weights = np.array(self.caps) / sum(self.caps)
        return self.permnos, self.returns, self.caps, self.market_weights
    
    def means_historical(self, date, lookback_window):
        self.permnos, self.returns, self.caps, self.market_weights = self.permnos_returns_caps_weights(date, lookback_window)
        self.returns = np.matrix(self.returns)
        self.rows, self.cols = self.returns.shape
        # calculate returns
        self.exp_returns = np.array([])
        for r in range(self.rows):
            self.exp_returns = np.append(self.exp_returns, np.mean(self.returns[r]))
        # calculate covariances
        self.covars = np.cov(self.returns)
        self.correls = np.corrcoef(self.returns)
        self.exp_returns = (1 + self.exp_returns) ** 250 - 1  # 3months returns
        self.covars = self.covars * 250  # 3months covariances
        return self.permnos, self.market_weights, self.exp_returns, self.covars, self.correls
    
    def return_summary(self, date, lookback_window):
        self.permnos, self.market_weights, self.exp_returns, self.covars, self.correls = self.means_historical(date, lookback_window)
        self.mean_weight_overview = pd.DataFrame({'Return': self.exp_returns, 'Market Weight': self.market_weights}, index=self.permnos).T
        self.cov_overview = pd.DataFrame(self.covars, columns=self.permnos, index=self.permnos)
        self.cor_overview = pd.DataFrame(self.correls, columns=self.permnos, index=self.permnos)
        return self.mean_weight_overview, self.cov_overview, self.cor_overview