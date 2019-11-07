import numpy                as np
import scipy.optimize
import scipy                as sp
import scipy.stats          as stat
import matplotlib.pyplot    as plt
from numpy.linalg           import inv, pinv
from sklearn.covariance     import LedoitWolf
from statsmodels.tsa.api    import Holt

from numpy import matrix, array, zeros, empty, sqrt, ones, dot, append, mean, cov, transpose, linspace, eye
from numpy.linalg import inv, pinv

class MeanVariance:
    """
    The methods of this class are generally used as input to all optimization techniques presented in this project. 
    That is why I decided to collect them into a parent class for the optimization.
    """

    def __init__(self, rf, permnos, returns, rebal_period, mean_pred=None, var_pred='MLE'):
        self.rf             = rf
        self.permnos        = permnos
        self.returns        = np.asarray(returns)
        self.n_assets       = len(self.permnos)
        self.mean_pred      = mean_pred
        self.var_pred       = var_pred
        self.rebal_period   = rebal_period
        self.R              = self.mean_model()
        self.C              = self.var_model()

    def mean_model(self):
        if self.mean_pred is None:
            return (1+np.mean(self.returns, axis=0))**self.rebal_period-1
        elif self.mean_pred == 'MLE':
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

    def var_model(self):
        # Fix problem with transposed returns!
        var_returns = self.returns
        if self.var_pred is None:
            return np.cov(var_returns)*self.rebal_period
        elif self.var_pred == 'LW':
            return LedoitWolf().fit(np.matrix(var_returns)).covariance_*self.rebal_period
        elif self.var_pred == 'MLE':
            return np.cov(var_returns.T)*self.rebal_period

    def port_mean(self, W):
        return sum(self.R * W)

    def port_var(self, W):
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
    
    def min_variance(self):
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bound       = (0.0,1.0)
        bounds      = tuple(bound for asset in range(self.n_assets))
        result      = scipy.optimize.minimize(self.port_var, self.n_assets*[1./self.n_assets,], method='SLSQP', bounds=bounds, constraints=constraints)
        if not result.success:
            raise BaseException(result.message)
        return result.x
    
    def efficient_return(self, target):
        constraints = ({'type': 'eq', 'fun': lambda x: self.port_mean(x) - target},
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds      = tuple((0,1) for asset in range(self.n_assets))
        result      = scipy.optimize.minimize(self.port_var, self.n_assets*[1./self.n_assets,], method='SLSQP', bounds=bounds, constraints=constraints)
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
        return np.array(frontier_ret), np.array(frontier_var), W_tan, np.array(tan_ret), np.array(tan_var)

    def display_assets(self, color='blue'):
        plt.scatter([self.C[i, i] ** .5 for i in range(self.n_assets)], self.R, marker='x', color=color)
    
    def display_frontier(self, label=None, color='blue'):
        front_mean, front_var, _, tan_mean, tan_var = self.efficient_frontier()
        plt.plot(front_var**0.5, front_mean, label=label, color=color)  # draw efficient frontier
    
class BlackLitterman(MeanVariance):

    def __init__(self, rf, permnos, returns, rebal_period, market_weights, mean_pred=None, var_pred='LW'):
        MeanVariance.__init__(self, rf, permnos, returns, rebal_period, mean_pred=None, var_pred='LW')
        self.market_weights = market_weights
        # Market implied returns
        self.R              = (1+np.dot(np.dot(4, self.C), self.market_weights)+rf)**rebal_period-1
        # If no views are presented, realize the market portfolio
        self.mu_c           = self.R        
    
    def get_model_return(self, tau, P, O, q):
        self.mu_c   = dot(inv(inv(tau*self.C)+dot(dot(transpose(P),inv(O)),P)),(dot(inv(tau*self.C),transpose([self.R.T]))+dot(dot(transpose(P),inv(O)),q))).flatten()
        return self.mu_c

    def bl_port_mean(self, W):
        return sum(self.mu_c * W)

    def efficient_return_bl(self, target):
        constraints = ({'type': 'eq', 'fun': lambda x: self.bl_port_mean(x) - target},
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds      = tuple((0,1) for asset in range(self.n_assets))
        result      = scipy.optimize.minimize(self.port_var, self.n_assets*[1./self.n_assets,], method='SLSQP', bounds=bounds, constraints=constraints)
        if not result.success:
            raise BaseException(result.message)
        return result.x

    def efficient_frontier_bl(self):
        frontier_ret = []
        frontier_var = []
        sharpe_ratio = float("-inf")
        for r in np.linspace(min(self.mu_c), max(self.mu_c)):  
            W_opt = self.efficient_return_bl(r)
            frontier_ret.append(r)
            frontier_var.append(self.port_var(W_opt))
            if self.sharpe_ratio(W_opt)>sharpe_ratio:
                sharpe_ratio             = self.sharpe_ratio(W_opt)
                W_tan                    = W_opt
                tan_ret                  = self.bl_port_mean(W_tan)  
                tan_var                  = self.port_var(W_tan)         
        return np.array(frontier_ret), np.array(frontier_var), W_tan, np.array(tan_ret), np.array(tan_var)

    def display_assets_bl(self, color='blue'):
        plt.scatter([self.C[i, i] ** .5 for i in range(self.n_assets)], self.mu_c, marker='x', color=color)
    
    def display_frontier_bl(self, label=None, color='blue'):
        front_mean, front_var, _, tan_mean, tan_var = self.efficient_frontier_bl()
        plt.scatter(tan_var ** .5, tan_mean, marker='o', color=color)
        plt.plot(front_var**0.5, front_mean, label=label, color=color)