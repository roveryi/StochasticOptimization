import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.optimize import minimize
from scipy.stats import norm

import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


class BayesianOptimizer(object):
    def __init__(self, black_box_func, x_bound, num_ini, num_iter, kind, kappa, xi, printflag = True):
        self.eva_func = black_box_func # Black box function 
        self.x_bound = x_bound # Optimal domain
        self.num_iter = num_iter # Number of iterations
        self.num_ini = num_ini # Number of initial trial
        self.gp = GaussianProcessRegressor(random_state=1) # Gaussian process 
        self.acquisition = acquisition_func(kind, kappa, xi) # Acquisition function
        self.printflag = printflag
        
    def RunOptimizer(self):
        x, y = rand_evaluate(self.x_bound, self.eva_func, self.num_ini)
        self.gp.fit(x, y)
        x_max, y_max = x[y.argmax()], np.max(y)
        i = 0
        hist = []
        while i < self.num_iter:
            x_suggest = acq_max(x_max, self.x_bound, y_max, self.acquisition, self.gp, 10, 80)
            y_suggest = self.eva_func([x_suggest])
            x = np.append(x,x_suggest.reshape(1,x.shape[1]),axis = 0)
            y = np.append(y, y_suggest, axis = 0)
            x_max, y_max = x[y.argmax()], np.max(y)
            self.gp.fit(x,y)
            i += 1
            hist.append(y_max)

            if self.printflag == True and i%10 == 0:
                print('%i th iteration right now! Current optimal value is %.6f'%(i, y_max))
        return x_max, y_max, hist


def rand_evaluate(x_bound, black_box_func, num_step):
    # Generate num_step times uniformly distributed random number within the bound
    x_rand = np.random.uniform(x_bound[0], x_bound[1], size = (num_step, x_bound.shape[1]))
    # Evaluate the response at each generated x
    y_rand = black_box_func(x_rand)
    return x_rand, y_rand

def acq_max(x_max, x_bound, y_max, acquisition, gp, num_step, n_iter):
    x_hist, y_hist = [], []
    x_tries = np.random.uniform(x_bound[0], x_bound[1], size = (num_step, x_bound.shape[1]))
    ys = acquisition.utility(x_tries, gp=gp, y_max=y_max)
    x_max = x_tries[ys.argmax()]
    max_acq = ys.max()
    x_seeds = np.random.uniform(x_bound[0], x_bound[1], size=(n_iter, x_bound.shape[1]))
    
    for x_try in x_seeds:
        # Find the minimum of minus the acquisition function
        res = minimize(lambda x: -acquisition.utility(x.reshape(1, -1), gp = gp, y_max = y_max),
                       x_try.reshape(1, -1),
                       bounds=x_bound.T,
                       method="L-BFGS-B")

        # See if success
        if not res.success:
            continue

        # Store it if better than previous minimum(maximum).
        if max_acq is None or -res.fun[0] >= max_acq:
            x_max = res.x
            max_acq = -res.fun[0]

    # Clip output to make sure it lies within the bounds. Due to floating
    # point technicalities this is not always the case.
    return np.clip(x_max, x_bound[0], x_bound[1])

class acquisition_func(object):
    def __init__(self, kind, kappa, xi):

        self.kappa = kappa # If UCB is adopted, kappa is required
        self.xi = xi
        self.kind = kind

    def utility(self, x, gp, y_max):
        if self.kind == 'ucb':
            return self._ucb(x, gp, self.kappa)
        if self.kind == 'ei':
            return self._ei(x, gp, y_max, self.xi)
        if self.kind == 'poi':
            return self._poi(x, gp, y_max, self.xi)

    @staticmethod
    def _ucb(x, gp, kappa):
        mean, std = gp.predict(x, return_std=True)
        std = std.reshape([1,-1])
        return mean + kappa * std

    @staticmethod
    def _ei(x, gp, y_max, xi):
        mean, std = gp.predict(x, return_std=True)
        std = std.reshape([1,-1])
        z = (mean - y_max - xi)/std
        return (mean - y_max - xi) * norm.cdf(z) + std * norm.pdf(z)

    @staticmethod
    def _poi(x, gp, y_max, xi):
        mean, std = gp.predict(x, return_std=True)
        std = std.reshape([1,-1])
        z = (mean - y_max - xi)/std
        return norm.cdf(z)