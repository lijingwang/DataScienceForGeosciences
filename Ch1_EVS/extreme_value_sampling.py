import numpy as np
from scipy.special import kn
from scipy.integrate import quad
from scipy.optimize import root
from scipy.stats import lognorm

# Author: Lijing Wang, lijing52@stanford.edu, 2020
# Reference: https://royalsocietypublishing.org/doi/pdf/10.1098/rspa.1977.0041 
# Equation (4.7)


# pdf f(x)
def hyperbolic_pdf(phi, gamma, mu, delta):
    omega = 1/(1/phi + 1/gamma)
    kappa = np.sqrt(phi*gamma)
    pdf = lambda x: (omega/(delta*kappa*kn(1,delta*kappa)))*np.exp(-0.5*(phi+gamma)*np.sqrt(delta**2+(x-mu)**2)+0.5*(phi-gamma)*(x-mu))
    return pdf

# cdf F(x)
def hyperbolic_cdf(phi, gamma, mu, delta):
    pdf = hyperbolic_pdf(phi, gamma, mu, delta)
    def cdf(x):
        return quad(pdf, -np.inf, x)[0]
    return cdf

# solve F(x) - y = 0, x = F^{-1}(y)
def inverse_hyperbolic_cdf(y, phi, gamma, mu, delta):
    ## cdf: F(x)
    cdf = hyperbolic_cdf(phi, gamma, mu, delta)
    
    ## shifted_cdf: F(x)-y
    shifted_cdf = lambda x: cdf(x)-y
    
    ## solve F(x)-y = 0, get x
    solve_root = root(shifted_cdf, 0.0001).x[0]
    
    return solve_root


# log hyperbolic sampling 
# --------------------------------------------------------
# Main function, the (only) function you need to run
# --------------------------------------------------------
def log_hyperbolic_sampling(N, phi, gamma, mu, delta, seed = 0):
    '''
    log hyperbolic distribution sampling with CDF inversion 
    
    Args: 
        N: (int) the number of samples
        phi, gamma, mu, delta: (float) parameters from log hyperbolic distribution, Eq. (4.7)
        seed: (int) the random seed
    Output:
        x: (np.array) sampling_array with length N. 
    
    Note:  
        For N = 1000, the sampling should be finished around 10s. 
    '''
    
    ## set random seed, default = 0
    np.random.seed(seed)
    
    ## Uniform sampling 
    y = np.random.uniform(size = N)
    x = np.zeros(N)
    
    for i in range(N):
        x[i] = np.exp(inverse_hyperbolic_cdf(y[i], phi, gamma, mu, delta)) # s = exp(v)

    return x


def log_normal_sampling(N, mu, sigma, seed = 10):
    '''
    log normal sampling
    
    Args: 
        N: (int) the number of samples
        mu, delta: (float) mu, sigma for normal distribution after logarithm
        seed: (int) the random seed
    Output:
        x: (np.array) sampling_array with length N. 
    '''
    ## set random seed, default = 10
    np.random.seed(seed)
    x = lognorm.rvs(s = sigma, loc = 0,scale = np.exp(mu), size = N)
    
    return x