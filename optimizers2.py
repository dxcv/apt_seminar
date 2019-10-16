import numpy as np
import scipy.optimize
import scipy as sp
import scipy.stats as stat
import matplotlib.pyplot as plt
from numpy.linalg import inv, pinv
from sklearn.covariance import LedoitWolf



class Optimizer:
    """
    The methods of this class are generally used as input to all optimization techniques presented in this thesis. 
    That is why i decided to collect them into a parent class for the optimization.
    """

    def __init__(self, rf, permnos, returns, rebal_period, mean_pred=None):
        self.rf         = rf
        self.permnos    = permnos
        self.returns    = np.matrix(returns)
        self.n_assets   = len(self.permnos)
        self.R = (1+np.mean(returns, axis=0))**rebal_period-1
        self.C = LedoitWolf().fit(self.returns).covariance_*rebal_period       

        # if mean_pred=None:
        #     self.R = (1+np.mean(returns, axis=0))**rebal_period-1
        # else:
        #     self.R = []
        #     _, cols = np.shape(self.returns)
        #     for i 
        # self.C = LedoitWolf().fit(self.returns).covariance_*rebal_period
    
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

class Markowitz(Optimizer):
    
    def solve_frontier(self):
        def fitness(W, r):
            # For given level of return r, find weights which minimizes portfolio variance.
            mean, var = self.port_mean_var(W)
            # Big penalty for not meeting stated portfolio return effectively serves as optimization constraint
            penalty = 100 * abs(mean - r)  
            return var + penalty
        # Initialize frontier mean and var
        frontier_mean, frontier_var = [], []

        # Number of assets in the portfolio
        n_assets = len(self.R)  

        # Iterate through the range of returns on Y axis
        for r in np.linspace(min(self.R), max(self.R), num=20):  
            # start optimization with equal weights
            W = np.ones([n_assets]) / n_assets  
            # Initialize boundaries for portfolio weights: 0,1
            b_ = [(0, 1) for i in range(n_assets)]
            # Sum of weights must be 100%
            c_ = ({'type': 'eq', 'fun': lambda W: sum(W) - 1.})
            optimized = scipy.optimize.minimize(fitness, W, r, method='SLSQP', constraints=c_, bounds=b_)
            if not optimized.success:
                raise BaseException(optimized.message)
            # add point to efficient frontier [x,y] = [optimized.x, r]
            frontier_mean.append(r)
            frontier_var.append(self.port_var(optimized.x))
        return np.array(frontier_mean), np.array(frontier_var)
    
    def solve_weights(self):
    # Given risk-free rate, assets returns and covariances, this function calculates
    # weights of tangency portfolio with respect to sharpe ratio maximization

        # Number of assets in the portfolio
        n_assets = len(self.R) 
        # start optimization with equal weights
        W = np.ones([n_assets]) / n_assets  
        # Initialize boundaries for portfolio weights: 0,1 
        b_ = [(0, 1) for i in range(n_assets)] 
        # Sum of weights must be 100%
        c_ = ({'type': 'eq', 'fun': lambda W: sum(W) - 1.})  
        optimized = scipy.optimize.minimize(self.inverse_sharpe_ratio, W, method='SLSQP', constraints=c_, bounds=b_)
        if not optimized.success: 
            # raise BaseException(optimized.message)
            # If optimization didn't work a monte carlo simulation steps in
            return self.solve_weights_mc()
        return optimized.x
    
    def solve_weights_ls(self):
    # Given risk-free rate, assets returns and covariances, this function calculates
    # weights of tangency portfolio with respect to sharpe ratio maximization

        # Number of assets in the portfolio
        n_assets = len(self.R) 
        # start optimization with equal weights
        W = np.ones([n_assets]) / n_assets  
        # Initialize boundaries for portfolio weights: 0,1 
        b_ = [(-1, 2) for i in range(n_assets)] 
        # Sum of weights must be 100%
        c_ = ({'type': 'eq', 'fun': lambda W: sum(W) - 1.})  
        optimized = scipy.optimize.minimize(self.inverse_sharpe_ratio, W, method='SLSQP', constraints=c_, bounds=b_)
        if not optimized.success: 
            # raise BaseException(optimized.message)
            # If optimization didn't work a monte carlo simulation steps in
            return self.solve_weights_mc()
        return optimized.x
    
    
    def solve_weights_mc(self, num_portfolios=10000):
        portfolio_weights = np.zeros(len(self.R))
        sharpe_ratio = 0
        for i in range(num_portfolios):
            np.random.seed(i)
            weights = np.random.random(len(self.R))
            weights /= np.sum(weights)
            if self.sharpe_ratio(W=weights) > sharpe_ratio:
                portfolio_weights = weights
        return portfolio_weights
                             
    def optimize_frontier(self):
        W = self.solve_weights()
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