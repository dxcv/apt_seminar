import numpy as np
import scipy.optimize
import scipy as sp
import scipy.stats as stat
import matplotlib.pyplot as plt
from numpy.linalg import inv, pinv
import random

class Markowitz:
    
    def __init__(self, rf, permnos, market_weights, exp_returns, covars):
        self.rf = rf
        self.permnos = permnos
        self.market_weights = market_weights
        self.R = exp_returns
        self.C = covars
        self.n_assets = len(self.R)
        
    # Calculates portfolio mean return
    def port_mean(self, W):
        return sum(self.R * W)

    # Calculates portfolio variance of returns
    def port_var(self, W):
        # Return W*C*W
        return np.dot(np.dot(W, self.C), W)
        
    # Combination of the two functions above - mean and variance of returns calculation
    def port_mean_var(self, W):
        return self.port_mean(W), self.port_var(W)
    
    def solve_frontier(self):
    # Given risk-free rate, assets returns and covariances, this function calculates
    # mean-variance frontier and returns its [x,y] points in two arrays
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
        def fitness(W):
            # Calculate mean/variance of the portfolio
            mean, var = self.port_mean_var(W)  
            # Utility = Sharpe ratio
            util = (mean - self.rf) / np.sqrt(var)  
            # Maximize the utility, minimize its inverse value
            return 1 / util  
        # Number of assets in the portfolio
        n_assets = len(self.R) 
        # start optimization with equal weights
        W = np.ones([n_assets]) / n_assets  
        # Initialize boundaries for portfolio weights: 0,1
        b_ = [(0, 1) for i in range(n_assets)] 
        # Sum of weights must be 100%
        c_ = ({'type': 'eq', 'fun': lambda W: sum(W) - 1.})  
        optimized = scipy.optimize.minimize(fitness, W, method='SLSQP', constraints=c_, bounds=b_)
        if not optimized.success: 
            raise BaseException(optimized.message)
        return optimized.x
            
    def optimize_frontier(self):
        W = self.solve_weights()
        # calculate tangency portfolio
        tan_mean, tan_var = self.port_mean_var(W)  
        # calculate efficient frontier
        front_mean, front_var = self.solve_frontier()  
        # Weights, Tangency portfolio asset means and variances, Efficient frontier means and variances
        return W, tan_mean, tan_var, front_mean, front_var  
    
    def display_assets(self, color='black'):
        # draw assets
        plt.scatter([self.C[i, i] ** .5 for i in range(self.n_assets)], self.R, marker='x', color=color), grid(True)
        for i in range(self.n_assets): 
            # draw labels
            text(self.C[i, i] ** .5, self.R[i], '  %s' % self.permnos[i], verticalalignment='center', color=color) 

    # def display_frontier(self, label=None, color='black'):
    #     W, tan_mean, tan_var, front_mean, front_var = self.optimize_frontier()
    #     text(tan_var ** .5, tan_mean, '   tangent', verticalalignment='center', color=color)
    #     scatter(tan_var ** .5, tan_mean, marker='o', color=color), grid(True)
    #     plot(front_var ** .5, front_mean, label=label, color=color), grid(True)  # draw efficient frontier

    
class BlackLitterman:
    
    def __init__(self, rf, permnos, market_weights, exp_returns, covars):
        self.rf = rf
        self.permnos = permnos
        self.market_weights = market_weights
        self.R = exp_returns
        self.C = covars
        self.n_assets = len(self.R)
        
    def port_mean_hist(self, W):
        return sum(self.R * W)
        
    def historical_risk_aversion(self):
        return (self.port_mean_hist(self.market_weights) - self.rf)/self.port_var(self.market_weights)
        
    def implied_returns(self):
        lmb = self.historical_risk_aversion()
        # Calculate equilibrium excess returns
        Pi_m = np.dot(np.dot(lmb, self.C), self.market_weights)
        return Pi_m+self.rf
    
    def port_mean(self, W):
        implR = self.implied_returns()
        return sum(implR * W)

    def port_var(self, W):
        #return W*C*W
        return np.dot(np.dot(W, self.C), W)
        
    def port_mean_var(self, W):
        return self.port_mean(W), self.port_var(W)
    
    def solve_frontier(self):
    # Given risk-free rate, assets returns and covariances, this function calculates
    # mean-variance frontier and returns its [x,y] points in two arrays
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

        # Get implied returns
        self.implR = self.implied_returns()
        
        # Iterate through the range of returns on Y axis
        for r in np.linspace(min(self.implR), max(self.implR), num=20):  
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
        def fitness(W):
            # Calculate mean/variance of the portfolio
            mean, var = self.port_mean_var(W)  
            # Utility = Sharpe ratio
            util = (mean - self.rf) / np.sqrt(var)  
            # Maximize the utility, minimize its inverse value
            return 1 / util  
        # Number of assets in the portfolio
        n_assets = len(self.R) 
        # start optimization with equal weights
        W = np.ones([n_assets]) / n_assets  
        # Initialize boundaries for portfolio weights: 0,1
        b_ = [(0, 1) for i in range(n_assets)] 
        # Sum of weights must be 100%
        c_ = ({'type': 'eq', 'fun': lambda W: sum(W) - 1.})  
        optimized = scipy.optimize.minimize(fitness, W, method='SLSQP', constraints=c_, bounds=b_)
        if not optimized.success: 
            raise BaseException(optimized.message)
        return optimized.x
            
    def optimize_frontier(self):
        W = self.solve_weights()
        # calculate tangency portfolio
        tan_mean, tan_var = self.port_mean_var(W)  
        # calculate efficient frontier
        front_mean, front_var = self.solve_frontier()  
        # Weights, Tangency portfolio asset means and variances, Efficient frontier means and variances
        return W, tan_mean, tan_var, front_mean, front_var  
    
    # def display_assets(self, color='black'):
    #     # draw assets
    #     scatter([self.C[i, i] ** .5 for i in range(self.n_assets)], self.implR, marker='x', color=color), grid(True)
    #     for i in range(self.n_assets): 
    #         # draw labels
    #         text(self.C[i, i] ** .5, self.implR[i], '  %s' % self.permnos[i], verticalalignment='center', color=color) 

    # def display_frontier(self, label=None, color='black'):
    #     W, tan_mean, tan_var, front_mean, front_var = self.optimize_frontier()
    #     text(tan_var ** .5, tan_mean, '   tangent', verticalalignment='center', color=color)
    #     scatter(tan_var ** .5, tan_mean, marker='o', color=color), grid(True)
    #     plot(front_var ** .5, front_mean, label=label, color=color), grid(True)  # draw efficient frontier