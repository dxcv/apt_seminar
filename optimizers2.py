import numpy as np
import scipy.optimize
import scipy as sp
import scipy.stats as stat
import matplotlib.pyplot as plt
from numpy.linalg import inv, pinv
from sklearn.covariance import LedoitWolf
from statsmodels.tsa.api import Holt

class Optimizer:
    """
    The methods of this class are generally used as input to all optimization techniques presented in this project. 
    That is why I decided to collect them into a parent class for the optimization.
    """

    def __init__(self, rf, permnos, returns, rebal_period, mean_pred=None):
        self.rf             = rf
        self.permnos        = permnos
        self.returns        = returns
        self.n_assets       = len(self.permnos)
        self.mean_pred      = mean_pred
        self.rebal_period   = rebal_period
        self.R              = self.holt()
        self.C = LedoitWolf().fit(np.matrix(self.returns)).covariance_*rebal_period

    def holt(self):
        if self.mean_pred is None:
            return (1+np.mean(self.returns, axis=0))**self.rebal_period-1
        elif self.mean_pred == 'Holt':
            temp = []
            _, cols = np.shape(np.matrix(self.returns))
            for i in range(cols):
                asset_return    = np.cumprod(1 + np.matrix(self.returns)[:,i].A1)
                model           = Holt(asset_return, exponential=True).fit(smoothing_level=0.8, smoothing_slope=0.2, optimized=False)
                pred            = model.forecast(self.rebal_period)[-1]/asset_return[-1]-1
                temp.append(pred)
            return np.asarray(temp)

    def port_mean(self, W):
        return sum(self.R * W)

    def port_var(self, W):
        # Return W*C*W
        return np.dot(np.dot(W, self.C), W)
        
    def port_mean_var(self, W):
        return self.port_mean(W), self.port_var(W)

    def sharpe_ratio(self, W):
        mean, var = self.port_mean_var(W)  
        return (mean - self.rf) / np.sqrt(var)  

    def inverse_sharpe_ratio(self, W):
        return 1/self.sharpe_ratio(W)

    def fitness(self, W, r):
        # For given level of return r, find weights which minimizes portfolio variance.
        mean, var = self.port_mean_var(W)
        # Big penalty for not meeting stated portfolio return effectively serves as optimization constraint
        penalty = 100 * abs(mean - r)  
        return var + penalty

    def n_assets(self):
        return len(self.R) 


class Markowitz(Optimizer):
    
    def solve_frontier(self):

        # Initialize frontier mean and var
        frontier_mean, frontier_var = [], []

        # Iterate through the range of returns on Y axis
        for r in np.linspace(min(self.R), max(self.R), num=20):  
            # start optimization with equal weights
            W = np.ones([self.n_assets]) / self.n_assets  
            # Initialize boundaries for portfolio weights: 0,1
            b_ = [(0, 1) for i in range(self.n_assets)]
            # Sum of weights must be 100%
            c_ = ({'type': 'eq', 'fun': lambda W: sum(W) - 1.})
            optimized = scipy.optimize.minimize(self.fitness, W, r, method='SLSQP', constraints=c_, bounds=b_)
            if not optimized.success:
                raise BaseException(optimized.message)
            # add point to efficient frontier [x,y] = [optimized.x, r]
            frontier_mean.append(r)
            frontier_var.append(self.port_var(optimized.x))
        return np.array(frontier_mean), np.array(frontier_var)
    
    def solve_tangency_weights(self):
        # Initialize sharpe ratio
        sharpe_ratio = -2

        # Iterate through the range of returns on Y axis
        for r in np.linspace(min(self.R), max(self.R), num=20):  
            # start optimization with equal weights
            W = np.ones([self.n_assets]) / self.n_assets  
            # Initialize boundaries for portfolio weights: 0,1
            b_ = [(.25, .75) for i in range(self.n_assets)]
            # Sum of weights must be 100%
            c_ = ({'type': 'eq', 'fun': lambda W: sum(W) - 1.})
            optimized = scipy.optimize.minimize(self.fitness, W, r, method='SLSQP', constraints=c_, bounds=b_)
            if not optimized.success:
                raise BaseException(optimized.message)
            elif self.sharpe_ratio(optimized.x)>sharpe_ratio:
                sharpe_ratio = self.sharpe_ratio(optimized.x)
                W = optimized.x
        return W
                             
    def optimize_frontier(self):
        W = self.solve_tangency_weights()
        # calculate tangency portfolio
        tan_mean, tan_var = self.port_mean_var(W)  
        # calculate efficient frontier
        front_mean, front_var = self.solve_frontier()  
        # Weights, Tangency portfolio asset means and variances, Efficient frontier means and variances
        return W, tan_mean, tan_var, front_mean, front_var  
    
    def display_assets(self, color='blue'):
        # draw assets
        plt.scatter([self.C[i, i] ** .5 for i in range(self.n_assets)], self.R, marker='x', color=color)
        #plt.show()
        # for i in range(self.n_assets): 
        #     # draw labels
        #     text(self.C[i, i] ** .5, self.R[i], '  %s' % self.permnos[i], verticalalignment='center', color=color) 

    def display_frontier(self, label=None, color='blue'):
        _, tan_mean, tan_var, front_mean, front_var = self.optimize_frontier()
        #text(tan_var ** .5, tan_mean, '   tangent', verticalalignment='center', color=color)
        plt.scatter(tan_var ** .5, tan_mean, marker='o', color=color)
        plt.plot(front_var ** .5, front_mean, label=label, color=color)  # draw efficient frontier