# CVXOPT
#%%
# Generate data for long only portfolio optimization.
import numpy as np
np.random.seed(1)
n       = 10
R       = np.abs(np.random.randn(n, 1))
Sigma   = np.random.randn(n, n)
C       = Sigma.T.dot(Sigma)
rf      = 0.5

#%%
def port_mean(R, W):
    return np.sum(R * W)

def port_var(C, W):
    # Return W*C*W
    return np.dot(np.dot(W, C), W)
    
def port_mean_var(W):
    return port_mean(R, W), port_var(C, W)

def sharpe_ratio(W, rf, R, C):
    mean, var = port_mean_var(W)  
    return (mean - rf) / np.sqrt(var)  

def inverse_sharpe_ratio(W):
    return 1/sharpe_ratio(W)


#%%
W = np.ones([n]) / n

port_var(C=C, W=W)
port_mean(R=R, W=W)
sharpe_ratio(W=W, rf=rf, R=R, C=C)

#%%


# Long only portfolio optimization.
from cvxpy import *
w       = Variable(n)
ret     = R.T*w
risk    = quad_form(w, C)
prob    = Problem(Maximize(ret/risk), [sum_entries(w) == 1, w >= 0])


#%%

# Compute trade-off curve.
SAMPLES = 100
risk_data = np.zeros(SAMPLES)
ret_data = np.zeros(SAMPLES)
gamma_vals = np.logspace(-2, 3, num=SAMPLES)
for i in range(SAMPLES):
    gamma.value = gamma_vals[i]
    prob.solve()
    risk_data[i] = sqrt(risk).value
    ret_data[i] = ret.value


#%%
# Plot long only trade-off curve.
import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format = 'svg'

markers_on = [29, 40]
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(risk_data, ret_data, 'g-')
for marker in markers_on:
    plt.plot(risk_data[marker], ret_data[marker], 'bs')
    ax.annotate(r"$\gamma = %.2f$" % gamma_vals[marker], xy=(risk_data[marker]+.08, ret_data[marker]-.03))
for i in range(n):
    plt.plot(sqrt(Sigma[i,i]).value, mu[i], 'ro')
plt.xlabel('Standard deviation')
plt.ylabel('Return')
plt.show()
