import numpy                as np
import scipy.optimize
import scipy                as sp
import scipy.stats          as stat
import matplotlib.pyplot    as plt
from numpy.linalg           import inv, pinv
from sklearn.covariance     import LedoitWolf
from statsmodels.tsa.api    import Holt

class MeanVariance:
    """
    The methods of this class are generally used as input to all optimization techniques presented in this project. 
    That is why I decided to collect them into a parent class for the optimization.
    """

    def __init__(self, rf, permnos, returns, rebal_period, mean_pred=None):
        self.rf             = rf
        self.permnos        = permnos
        self.returns        = np.asarray(returns).T
        self.n_assets       = len(self.permnos)
        self.mean_pred      = mean_pred
        self.rebal_period   = rebal_period
        self.R              = self.holt()
        self.C = LedoitWolf().fit(np.matrix(self.returns)).covariance_*rebal_period
        # self.C              = np.cov(np.asarray(self.returns).T)

    def holt(self):
        if self.mean_pred is None:
            return (1+np.mean(self.returns, axis=0))**self.rebal_period-1
        elif self.mean_pred == 'Holt':
            temp = []
            _, cols             = np.shape(np.matrix(self.returns))
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
        mean, var   = self.port_mean_var(W)
        penalty     = 100*abs(mean - r)  
        return var + penalty

    def solve_frontier(self):
        frontier_mean, frontier_var = [], []
        for r in np.linspace(min(self.R), max(self.R), num=20):  
            W   = np.ones([self.n_assets]) / self.n_assets  
            b_  = [(0, 1) for i in range(self.n_assets)]
            c_  = ({'type': 'eq', 'fun': lambda W: sum(W) - 1.})
            optimized = scipy.optimize.minimize(self.fitness, W, r, method='SLSQP', constraints=c_, bounds=b_)
            if not optimized.success:
                raise BaseException(optimized.message)
            # add point to efficient frontier [x,y] = [optimized.x, r]
            frontier_mean.append(r)
            frontier_var.append(self.port_var(optimized.x))
        return np.array(frontier_mean), np.array(frontier_var)
    
    def solve_tangency_weights(self):
        sharpe_ratio = -2
        # Iterate through the range of returns on Y axis
        for r in np.linspace(min(self.R), max(self.R)):  
            W   = np.ones([self.n_assets]) / self.n_assets  
            b_  = [(0, 1) for i in range(self.n_assets)]
            c_  = ({'type': 'eq', 'fun': lambda W: sum(W) - 1.})
            optimized = scipy.optimize.minimize(self.fitness, W, r, method='SLSQP', constraints=c_, bounds=b_)
            if not optimized.success:
                raise BaseException(optimized.message)
            elif self.sharpe_ratio(optimized.x)>sharpe_ratio:
                sharpe_ratio = self.sharpe_ratio(optimized.x)
                W = optimized.x
        return W
                             
    def optimize_frontier(self):
        W                       = self.solve_tangency_weights()
        tan_mean, tan_var       = self.port_mean_var(W)  
        front_mean, front_var   = self.solve_frontier()  
        return W, tan_mean, tan_var, front_mean, front_var  
    
    def min_variance(self):
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bound = (0.0,1.0)
        bounds = tuple(bound for asset in range(self.n_assets))
        result = scipy.optimize.minimize(self.port_var, self.n_assets*[1./self.n_assets,], method='SLSQP', bounds=bounds, constraints=constraints)
        if not result.success:
            raise BaseException(result.message)
        return result.x
    
    def efficient_return(self, target):
        constraints = ({'type': 'eq', 'fun': lambda x: self.port_mean(x) - target},
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0,1) for asset in range(self.n_assets))
        result = scipy.optimize.minimize(self.port_var, self.n_assets*[1./self.n_assets,], method='SLSQP', bounds=bounds, constraints=constraints)
        if not result.success:
            raise BaseException(result.message)
        return result.x

    def efficient_frontier(self):
        frontier_ret = []
        frontier_var = []
        sharpe_ratio = float("-inf")
        for r in np.linspace(min(self.R), max(self.R)):  
            W_opt = self.efficient_return(r)
            frontier_ret.append(r)
            frontier_var.append(self.port_var(W_opt))
            if self.sharpe_ratio(W_opt)>sharpe_ratio:
                sharpe_ratio             = self.sharpe_ratio(W_opt)
                W_tan                    = W_opt
                tan_ret, tan_var         = self.port_mean_var(W_tan)  
                # print('Vol:', np.sqrt(tan_var), 'SR:', sharpe_ratio)             
        return np.array(frontier_ret), np.array(frontier_var), W_tan, np.array(tan_ret), np.array(tan_var)

    def display_assets(self, color='blue'):
        plt.scatter([self.C[i, i] ** .5 for i in range(self.n_assets)], self.R, marker='x', color=color)
        # for i in range(self.n_assets): 
        #     plt.text(self.C[i, i] ** .5, self.R[i], '  %s' % self.permnos[i], color=color)
    
    def display_frontier(self, label=None, color='blue'):
        # _, tan_mean, tan_var, front_mean, front_var = self.optimize_frontier()
        front_mean, front_var, _, tan_mean, tan_var = self.efficient_frontier()
        # text(tan_var ** .5, tan_mean, '   tangent', verticalalignment='center', color=color)
        # plt.scatter(tan_var**0.5, tan_mean, marker='o', color=color)
        plt.plot(front_var**0.5, front_mean, label=label, color=color)  # draw efficient frontier
    
class BlackLitterman(MeanVariance):

    def __init__(self, rf, permnos, returns, rebal_period, market_weights, mean_pred=None):
        MeanVariance.__init__(self, rf, permnos, returns, rebal_period, mean_pred=None)
        self.market_weights = market_weights
        self.R              = (1+np.dot(np.dot(3, self.C), self.market_weights)+rf)**rebal_period-1

