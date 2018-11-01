#!/usr/bin/python2

from numpy import *
from copy import deepcopy
import cma
import lcmaes

import contextlib
import io
import sys
import random
import sys

from scipy.linalg import solve_triangular, inv

# Hack for avoiding the lcmaes library to print.
@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = io.BytesIO()
    yield
    sys.stdout = save_stdout

class RBF():
    def __init__(self, length=1, sigma=1, noise=1e-10, df=None):
        if df:
            self.df = df
        else:
            self.df = lambda x_1, x_2: linalg.norm(x_1 - x_2)
        self.set_hps(length, sigma)
        self.noise = noise
    
    def set_hps(self, length, sigma, noise=1e-10):
        self.ls = power(length, 2.)
        self.ss = power(sigma, 2.)
        self.noise = noise
    
    def get_hps(self):
        return self.ls, self.ss, self.noise
    
    def get_gram(self, samples):
        if type(samples) == matrix:
            m = samples.shape[0]
        else:
            m = len(samples)
        
        # TODO: Optimize using several threads
        gram = zeros((m, m))
        for i in xrange(m):
            gram[i, i] = power(self.df(samples[i], samples[i]), 2.)
        for i in xrange(m):
            for j in xrange(i):
                gram[i, j] = power(self.df(samples[i], samples[j]), 2.)
                gram[j, i] = gram[i, j]
        gram /= gram.max() # normalize 
        return gram
    
    # k = ss*exp(-(1/2*ls) .* m) + eye(size(m))*ns
    def gram_to_kernel(self, gram):
        # return exp(log(self.ss) + gram * (-0.5/self.ls)) + \
        #     eye(gram.shape[0]) * self.noise
        return self.ss * exp((-0.5/self.ls) * gram) + \
            eye(gram.shape[0]) * self.noise

    def __call__(self, x_1, x_2):
        d = power(self.df(x_1, x_2), 2.)
        return self.ss * (exp(-0.5 * d / self.ls))

class GaussianProcessModel():
    def __init__(self, kernel_function, dim_in, dim_out, noise=0, optimize_noise=False):
        self.kf = kernel_function
        self.dim_in = dim_in
        self.dim_out = dim_out
        
        self.samples = []
        self.observations = matrix([])
        self.kernel = matrix([[]])
        
        self.observations_mean = matrix([])
        self.zero_meaned_obs = matrix([])
        self.matrixL = matrix([[]])
        self.alpha = matrix([[]])
        self.K_inv = matrix([[]])
        self.gram = matrix([[]])
        
        self.optimize_noise = optimize_noise
        
        self.negative_variance_warning = False
        self.negative_variance_warning_count = 0
    
    def set(self, samples, observations):
        self.samples = samples
        if type(observations) == matrix:
            self.observations = observations
        else:
            self.observations = matrix(observations).T
        self.observations_mean = mean(observations, axis=0)
        
        # Get zero meaned observations
        self.zero_meaned_obs = self.observations - self.observations_mean
        self.gram = self.kf.get_gram(self.samples)
        
    def compute(self, samples, observations, compute_kernel=True):
        
        # assert samples.shape[0] != 0
        # assert observations.shape[0] != 0
        # assert samples.shape[0] == observations.shape[0]
        
        self.samples = samples
        if type(observations) == matrix:
            self.observations = observations
        else:
            self.observations = matrix(observations).T
        self.observations_mean = mean(observations, axis=0)
        
        # Get zero meaned observations
        self.zero_meaned_obs = self.observations - self.observations_mean
        
        if compute_kernel:
            self.__compute_full_kernel()
            
    def recompute(self):
        self.__recompute_kernel()
    
    def predict(self, x):
        k = self.__compute_k(x)
        mean = self.__mean(x, k)
        std = self.__std(x, k)
        
        if mean.size == 1:
            mean = mean[0]
        if std.size == 1:
            std = std[0]
        return (mean, std)
    
    def optimize(self, quiet=True, bounded=False, fevals=1000, compute_gram=False):
        
        # Wrap the likelihood function to optimize, take into account it could
        # be run in parallel.
        self.count = 0
        if quiet:
            print "[GP] Optimizing...",
        def likelihood(gpm, args, n):
            self.count += 1
            if quiet:
                if self.count % 100 == 0:
                    print str(self.count) + " ",
                    sys.stdout.flush()
            
            args = exp(args)
            gpm = deepcopy(gpm)
            gpm.kf.set_hps(*args)
            try:
                if compute_gram:
                    gpm.gram = gpm.kf.get_gram(gpm.samples)
                gpm.recompute()
            except linalg.linalg.LinAlgError:
                return inf
            return float(-gpm.get_log_likelihood())
        
        # Define the starting point for the arguments
        init_args = [1, 1] # length, sigma
        if self.optimize_noise:
            init_args += [1e-3]
        init_args = map(log, init_args)
        
        # Define the starting point of the cames variables.
        lambda_ = 10    # lambda is a reserved keyword in python, using lambda_ instead.
        seed = 0        # 0 for seed auto-generated within the lib.
        sigma = 1
        lbounds = log([0.001, 0.001, 1e-5]) # l, s, n
        ubounds = log([10, 10, 0.1])
        
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
        p.set_restarts(3)
        p.set_max_fevals(fevals)
        p.set_elitism(3)
        p.set_mt_feval(True) # NOTE: For some reason this is not working.
        
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
        hparams = exp(bx)        
        self.kf.set_hps(*hparams)
        if compute_gram:
            self.gram = self.kf.get_gram(self.samples)
        self.recompute()
        
        print "[GP] Hyperparameters: ", hparams
    
    def get_log_likelihood(self):
        # --- cholesky ---
        # see:
        # http://xcorr.net/2008/06/11/log-determinant-of-positive-definite-matrices-in-matlab/
        # L = chol(A);
        # logdetA = 2*sum(log(diag(L)));
        m = self.zero_meaned_obs.shape[0]
        det = 2 * sum(log(self.matrixL.diagonal()))
        a = sum(dot(self.zero_meaned_obs.T, self.alpha))
        return -0.5 * a - 0.5 * det - 0.5 * m * log(2 * pi)
    
    def __compute_full_kernel(self):
        self.gram = self.kf.get_gram(self.samples)
        self.__recompute_kernel()
        
    def __recompute_kernel(self):
        self.kernel = self.kf.gram_to_kernel(self.gram)
        
        if False:
            eigenvalues, eigenvectors = linalg.eig(self.kernel)
            if False:
                for i in range(len(eigenvalues)):
                    eigenvalues[i] = max(1e-10, eigenvalues[i])
            
            value = 0
            if True:
                neg_eigenvalues = filter(lambda v: v < 0, list(eigenvalues))
                if neg_eigenvalues:
                    value = abs(min(neg_eigenvalues)) + 1e-10
            
            RK = dot(eigenvectors, (eye(eigenvalues.shape[0]) * value + eye(eigenvalues.shape[0]) * eigenvalues).dot(linalg.inv(eigenvectors)))
            self.kernel = RK
        
        
        # Get decomposition and compute alpha
        self.matrixL = linalg.cholesky(self.kernel)
        self.__compute_alpha()
        self.K_inv = matrix([[]])
        
        # Enable the negative variance warning
        self.negative_variance_warning = False
        self.negative_variance_warning_count = 0
        
    def __compute_alpha(self):
        # alpha = K^{-1} * self.zero_meaned_obs
        self.alpha = self.__solve_with_LLT(self.matrixL, self.zero_meaned_obs)
        
    def __solve_with_LLT(self, L, b):
        y = linalg.lstsq(L, b)[0]
        return linalg.lstsq(L.T, y)[0]
    
    def __mean(self, x, k):
        result = dot(k.T, self.alpha) + self.observations_mean        
        return self.__to_vector(result)
    
    def __std(self, x, k):        
        if self.K_inv.size == 0:
            L_inv = solve_triangular(self.matrixL.T, eye(self.matrixL.shape[0]))
            # L_inv = inv(self.matrixL)
            self.K_inv = L_inv.dot(L_inv.T)
        
        y_var = (self.kf(x, x) - dot(k.T, self.K_inv).dot(k))[0]
        if y_var[0] < 0:
            if not self.negative_variance_warning:
                print "\n[WARN]\n[WARN]\n[WARN] Predicted variance smaller than 0: " + str(y_var[0]) + "\n[WARN]\n[WARN]\n"
                self.negative_variance_warning = True
            self.negative_variance_warning_count += 1
            y_var = array([0])
        return sqrt(y_var)
        
    def __compute_k(self, x):
        if type(self.samples) == matrix:
            m = self.samples.shape[0]
        else:
            m = len(self.samples)
        k = zeros((m, 1))
        for i in range(m):
            k[i] = self.kf(self.samples[i], x)
            
        if self.K_inv.size == 0:
            L_inv = solve_triangular(self.matrixL.T, eye(self.matrixL.shape[0]))
            # L_inv = inv(self.matrixL)
            self.K_inv = L_inv.dot(L_inv.T)
        return k
    
    def __to_vector(self, m):
        return asarray(m).reshape(-1)
