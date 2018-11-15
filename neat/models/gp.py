#!/usr/bin/python2
from __future__ import division
from copy import deepcopy
import cma
import lcmaes

import contextlib
import io
import sys
import random
import sys
import numpy as np

from scipy.linalg import solve_triangular, inv, cholesky, lstsq

# Hack for avoiding the lcmaes library to print.
@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = io.BytesIO()
    yield
    sys.stdout = save_stdout

class RBF():
    def __init__(self, sigma=1, length=1, noise=1e-5, df=None):
        if df:
            self.df = df
        else:
            self.df = lambda x_1, x_2: np.linalg.norm(x_1 - x_2)
        self.set_hps(sigma, length, noise)
    
    def set_hps(self, sigma, length, noise=1e-5):
        self.ss = sigma
        self.ls = length
        self.noise = noise
    
    def get_hps(self):
        return self.ss, self.ls, self.noise
    
    def get_squared_gram(self, samples, normalize=False):
        if type(samples) == np.matrix:
            n = samples.shape[0]
        else:
            n = len(samples)
        
        # TODO: Optimize using several threads
        gram = np.zeros((n, n), dtype=np.float64)
        for i in xrange(n):
            gram[i, i] = np.float64(np.power(self.df(samples[i], samples[i]), 2.))
        for i in xrange(n):
            for j in xrange(i):
                gram[i, j] = np.float64(np.power(self.df(samples[i], samples[j]), 2.))
                gram[j, i] = gram[i, j]
        nvalue = 1.
        if normalize:
            nvalue = gram.max()
            gram /= gram.max() # normalize 
        return gram, nvalue
    
    def gram_to_kernel(self, gram):
        return self.ss * np.exp((-0.5/self.ls) * gram) + \
            np.eye(gram.shape[0]) * self.noise

    def __call__(self, x_1, x_2, normalize_by=1.):
        d = np.power(self.df(x_1, x_2), 2.)/normalize_by
        return self.ss * np.exp((-0.5/self.ls) * d)

class GaussianProcessModel():
    def __init__(self, kernel_function, dim_in, dim_out, noise=0, optimize_noise=False, normalize_gram=False):
        self.kf = kernel_function
        self.dim_in = dim_in
        self.dim_out = dim_out
        
        self.samples = []
        self.observations = np.matrix([], dtype=np.float64)
        self.kernel = np.matrix([[]], dtype=np.float64)
        
        self.observations_mean = np.matrix([], dtype=np.float64)
        self.zero_meaned_obs = np.matrix([], dtype=np.float64)
        self.matrixL = np.matrix([[]], dtype=np.float64)
        self.alpha = np.matrix([[]], dtype=np.float64)
        self.K_inv = np.matrix([[]], dtype=np.float64)
        self.squared_gram = np.matrix([[]], dtype=np.float64)
        
        self.optimize_noise = optimize_noise
        self.normalize_gram = normalize_gram
        self.gram_nvalue = 1.
        
        self.negative_variance_warning = False
        self.negative_variance_warning_count = 0
        self.hp_not_positive_definite_error_count = 0
        
    def compute(self, samples, observations, compute_kernel=True):
        """Set the samples and observations and compute the gram matrix and 
        kernel."""
        
        self.samples = samples
        if type(observations) == np.matrix:
            self.observations = observations
        else:
            self.observations = np.matrix(observations, dtype=np.float64).transpose()
        assert(self.observations.shape[1] == self.dim_out)
        self.observations_mean = np.mean(observations, axis=0)
        
        # Get zero meaned observations
        self.zero_meaned_obs = self.observations - self.observations_mean
        
        if compute_kernel:
            self.__compute_full_kernel()
            
    def recompute(self):
        """Compute the kernel without recomputing the gram matrix. Helps to
        save time when only the hiperparameters are being changed."""
        
        self.__recompute_kernel()
    
    def predict(self, x):
        if type(x) in [np.matrix, np.array, np.ndarray] and \
            ((len(x.shape) == 1 and self.dim_in != x.shape[0]) or \
            (len(x.shape) == 2 and 1 < x.shape[0])):
            means = []
            deviations = []
            for i in range(x.shape[0]):
                mu, std = self.predict(x[i])
                means.append(mu)
                deviations.append(std)
            means = np.array(means)
            deviations = np.array(deviations)
            return (means, deviations)
        
        k = self.__compute_k(x)
        mean = self.__mean(x, k)
        std = self.__std(x, k)
        
        # If the output dimension is just 1, return a tuple.
        if mean.size == 1:
            mean = mean[0]
        if std.size == 1:
            std = std[0]
            
        return (mean, std)
    
    def optimize(self, quiet=True, bounded=False, fevals=1000, pub=[1e5, 1e5, 0.1], plb=[1e-5, 1e-5, 1e-5], compute_gram=False):
        
        # Wrap the likelihood function to optimize, take into account it could
        # be run in parallel.
        self.count = 0
        self.min_args = None
        self.min_score = None
        self.min_gpm = None
        self.hp_not_positive_definite_error_count = 0
        if quiet:
            print "[GP] Optimizing...",
            
        def likelihood(gpm, args, n):
            self.count += 1
            if quiet:
                if self.count % 100 == 0:
                    print str(self.count) + " ",
                    sys.stdout.flush()
            
            args = np.power(np.exp(args), 2)
            gpm = deepcopy(gpm)
            gpm.kf.set_hps(*args)
            try:
                if compute_gram:
                    gpm.squared_gram, gpm.gram_nvalue = gpm.kf.get_squared_gram(gpm.samples, gpm.normalize_gram)
                gpm.recompute()
            except np.linalg.linalg.LinAlgError:
                # print args
                # print
                # print gpm.squared_gram
                # print
                # print gpm.kf.gram_to_kernel(gpm.squared_gram)
                self.hp_not_positive_definite_error_count += 1
                return np.inf
            
            score = -gpm.get_log_likelihood()
            if not self.min_score or score < self.min_score:
                self.min_score = score
                self.min_args = args
                self.min_gpm = gpm
            
            return float(score)
        
        # Define the starting point for the arguments
        init_args = [1, 1] # length, sigma
        if self.optimize_noise:
            init_args += [1e-3]
        init_args = map(np.log, init_args)
        
        # Define the starting point of the CMA-ES variables.
        lambda_ = 10    # lambda is a reserved keyword in python, using lambda_ instead.
        seed = 1        # 0 for seed auto-generated within the lib.
        sigma = 2
        ubounds = np.log(np.sqrt(pub))
        lbounds = np.log(np.sqrt(plb)) # l, s, n
        
        if bounded:
            geno = lcmaes.make_genopheno_pwqb(list(lbounds), list(ubounds), 3)
            p = lcmaes.make_parameters_pwqb(init_args, sigma, geno, lambda_, seed)
        else:
            p = lcmaes.make_simple_parameters(init_args, sigma, lambda_, seed)
        
        # Defines the parameters of the optimizer.
        p.set_quiet(quiet)
        p.set_mt_feval(1)
        p.set_tpa(2)
        p.set_str_algo("abipop")
        # p.set_str_algo("acmaes")
        p.set_restarts(5)
        p.set_max_fevals(fevals)
        p.set_elitism(3)
        
        # Copy the model in the case this is being run in several threads.
        gpm = deepcopy(self)
        ll = lambda args, n: likelihood(gpm, args, n) # bind context to func.
        objfunc = lcmaes.fitfunc_pbf.from_callable(ll);
        
        # Run the optimizer.
        if bounded:
            cmasols = lcmaes.pcmaes_pwqb(objfunc, p)
        else:
            cmasols = lcmaes.pcmaes(objfunc, p)
        
        if quiet:
            print
        
        # Recover the solution, set the hparams of the model and recompute.
        bcand = cmasols.best_candidate()
        bx = lcmaes.get_candidate_x(bcand)
        hparams = np.power(np.exp(bx), 2)
        if self.min_args is None:
            self.min_args = hparams
        self.kf.set_hps(*self.min_args)
        if compute_gram:
            self.squared_gram, self.gram_nvalue = self.kf.get_squared_gram(self.samples, self.normalize_gram)
        self.recompute()
        
        print "[GP] Hyperparameters: ", self.min_args
    
    def get_log_likelihood(self):
        # see:
        # http://xcorr.net/2008/06/11/log-determinant-of-positive-definite-matrices-in-matlab/
        m = self.zero_meaned_obs.shape[0]
        det = 2 * sum(np.log(self.matrixL.diagonal()))
        a = sum(np.dot(self.zero_meaned_obs.T, self.alpha))
        return (-0.5 * a - 0.5 * det - 0.5 * m * np.log(2 * np.pi))[0, 0]
    
    def __compute_full_kernel(self):
        self.squared_gram, self.gram_nvalue = self.kf.get_squared_gram(self.samples, self.normalize_gram)
        self.__recompute_kernel()
        
    def __recompute_kernel(self):
        self.kernel = self.kf.gram_to_kernel(self.squared_gram)
        
        # Get decomposition and compute alpha
        self.matrixL = np.linalg.cholesky(self.kernel)
        self.__compute_alpha()
        self.K_inv = np.matrix([[]], dtype=np.float64)
        
        # Enable the negative variance warning
        self.negative_variance_warning = False
        self.negative_variance_warning_count = 0
        
    def __compute_alpha(self):
        # alpha = K^{-1} * self.zero_meaned_obs
        self.alpha = self.__solve_with_LLT(self.matrixL, self.zero_meaned_obs)
    
    def __compute_inverse(self):
        if self.K_inv.size == 0:
            L_inv = solve_triangular(self.matrixL.T, np.eye(self.matrixL.shape[0]), check_finite=False)
            self.K_inv = L_inv.dot(L_inv.T)
        
    def __solve_with_LLT(self, L, b):
        y = np.linalg.lstsq(L, b, rcond=None)[0]
        return np.linalg.lstsq(L.T, y, rcond=None)[0]
    
    def __mean(self, x, k):
        result = np.dot(k.T, self.alpha) + self.observations_mean
        return self.__to_vector(result)
    
    def __std(self, x, k):
        self.__compute_inverse()
        y_var = (self.kf(x, x, normalize_by=self.gram_nvalue) - np.dot(k.T, self.K_inv).dot(k))[0]
        if y_var[0] < 0:
            if not self.negative_variance_warning:                
                # print "\n[WARN]\n[WARN]\n[WARN] Predicted variance smaller than 0: " + str(y_var[0]) + "\n[WARN]\n[WARN]\n"
                self.negative_variance_warning = True
            self.negative_variance_warning_count += 1
            y_var = np.array([0])
        return np.sqrt(y_var)
        
    def __compute_k(self, x):
        if type(self.samples) == np.matrix:
            m = self.samples.shape[0]
        else:
            m = len(self.samples)
        k = np.zeros((m, 1))
        for i in range(m):
            k[i] = self.kf(self.samples[i], x, normalize_by=self.gram_nvalue)
        return k
    
    def __to_vector(self, m):
        return np.asarray(m).reshape(-1)
