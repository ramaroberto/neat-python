import sys
import unittest

# Import the local version of neat instead of the one in the library
import imp, sys
from os import path
local_neat_path = path.dirname(path.dirname(path.abspath(__file__)))
f, pathname, desc = imp.find_module("neat", [local_neat_path])
local_neat = imp.load_module("neat", f, pathname, desc)

from neat.surrogate import DefaultSurrogateModel as DefaultSurrogateModel
from neat.models.gp import RBF, GaussianProcessModel

from sklearn.gaussian_process.kernels import RBF as SKRBF, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor

import neat
import test_utils as tutils
import numpy as np
import numpy.testing as npt
import scipy as sp
import matplotlib.pyplot as plt

local_dir = path.dirname(__file__)
config_path = path.join(local_dir, 'test_configuration_surrogate')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path, neat.surrogate.DefaultSurrogateModel)

stagnation = config.stagnation_type(config.stagnation_config, None)
reproduction = config.reproduction_type(config.reproduction_config, None, 
    stagnation)

genomes = reproduction.create_new(config.genome_type, config.genome_config, 
    100).values()

def get_genomes_id(genomes):
    return map(lambda g: g.key, genomes)

class TestAddToTrainingComputation(unittest.TestCase):
    def test_add_1(self):
        surrogate = DefaultSurrogateModel(config, None)
        surrogate.add_to_training([genomes[0]])
        self.assertEqual(get_genomes_id(surrogate.training_set), 
            get_genomes_id([genomes[0]]))
    
    def test_add_multiple(self):
        surrogate = DefaultSurrogateModel(config, None)
        surrogate.add_to_training(genomes[0:3])
        self.assertEqual(get_genomes_id(surrogate.training_set), 
            get_genomes_id(genomes[0:3]))
    
    def test_add_over_the_limit(self):
        surrogate = DefaultSurrogateModel(config, None)
        surrogate.max_training_set_size = 20
        surrogate.add_to_training(genomes[0:30])
        self.assertEqual(get_genomes_id(surrogate.training_set), 
            get_genomes_id(genomes[10:30]))
    
    def test_add_when_reseted_1(self):
        surrogate = DefaultSurrogateModel(config, None)
        surrogate.add_to_training(genomes[0:3])
        
        surrogate.reset()
        surrogate.add_to_training([genomes[3]])
        self.assertEqual(get_genomes_id(surrogate.old_training_set), 
            get_genomes_id(genomes[0:3]))
        self.assertEqual(get_genomes_id(surrogate.training_set), 
            get_genomes_id([genomes[3]]))
    
    def test_add_when_reseted_mult(self):
        surrogate = DefaultSurrogateModel(config, None)
        surrogate.add_to_training(genomes[0:3])
        
        surrogate.reset()
        surrogate.add_to_training(genomes[3:6])
        self.assertEqual(get_genomes_id(surrogate.old_training_set), 
            get_genomes_id(genomes[0:3]))
        self.assertEqual(get_genomes_id(surrogate.training_set), 
            get_genomes_id(genomes[3:6]))
    
    def test_add_when_reseted_over_the_limit(self):
        surrogate = DefaultSurrogateModel(config, None)
        surrogate.max_training_set_size = 20
        
        surrogate.add_to_training(genomes[0:10])    # Adds 10 genomes
        surrogate.reset()
        surrogate.add_to_training(genomes[10:25])   # Adds 15 genomes
        
        self.assertEqual(get_genomes_id(surrogate.old_training_set), 
            get_genomes_id(genomes[5:10]))
        self.assertEqual(get_genomes_id(surrogate.training_set), 
            get_genomes_id(genomes[10:25]))
        
        for genome in genomes:
            genome.fitness = 0.0
            genome.real_fitness = 0.0
        surrogate.train(0, config)
        self.assertEqual(get_genomes_id(surrogate.old_training_set), [])
        self.assertEqual(get_genomes_id(surrogate.training_set), 
            get_genomes_id(genomes[5:25]))

class TestIsTrainingSetNew(unittest.TestCase):
    def test_simple(self):
        surrogate = DefaultSurrogateModel(config, None)
        surrogate.max_training_set_size = 20
        
        surrogate.add_to_training(genomes[0:10])    # Adds 10 genomes
        surrogate.reset()
        surrogate.add_to_training(genomes[10:25])   # Adds 15 genomes
        self.assertEqual(surrogate.is_training_set_new(), False)
        
        surrogate.add_to_training(genomes[25:29])   # Adds 4 genomes
        self.assertEqual(surrogate.is_training_set_new(), False)
        
        surrogate.add_to_training(genomes[29:30])   # Adds 1 genome
        self.assertEqual(surrogate.is_training_set_new(), True)

class TestRBFKernel(unittest.TestCase):
    def test_init(self):
        df = lambda x_1, x_2: np.linalg.norm(x_1 - x_2)
        params = [1., 2., 0.001, df]
        kernel = RBF(*params)
        self.assertEqual([kernel.ss, kernel.ls, kernel.noise, kernel.df], params)
        
    def test_set_get_hparameters(self):
        params = [1., 2., 0.001]
        kernel = RBF(*params)
        self.assertEqual([kernel.ss, kernel.ls, kernel.noise], params)
        
        params = [1., 2., 3.]
        kernel.set_hps(*params)
        self.assertEqual(list(kernel.get_hps()), params)
        
        params = [100., 200., 300.]
        kernel.set_hps(*params)
        self.assertEqual(list(kernel.get_hps()), params)

    def test_get_gram(self):
        df = lambda x_1, x_2: np.linalg.norm(x_1 - x_2)
        params = [0, 0, 0, df]
        kernel = RBF(*params)
        
        # Check with constant numbers
        samples = np.matrix(np.linspace(-5, 5, 11)).transpose()
        g1, nvalue = kernel.get_squared_gram(samples, normalize=False)
        g1n, nvalue = kernel.get_squared_gram(samples, normalize=True)
        should_be = np.power(
            np.matrix([[0.  , 1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00, 10.0],
                       [1.00, 0.  , 1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00],
                       [2.00, 1.00, 0.  , 1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00],
                       [3.00, 2.00, 1.00, 0.  , 1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00],
                       [4.00, 3.00, 2.00, 1.00, 0.  , 1.00, 2.00, 3.00, 4.00, 5.00, 6.00],
                       [5.00, 4.00, 3.00, 2.00, 1.00, 0.  , 1.00, 2.00, 3.00, 4.00, 5.00],
                       [6.00, 5.00, 4.00, 3.00, 2.00, 1.00, 0.  , 1.00, 2.00, 3.00, 4.00],
                       [7.00, 6.00, 5.00, 4.00, 3.00, 2.00, 1.00, 0.  , 1.00, 2.00, 3.00],
                       [8.00, 7.00, 6.00, 5.00, 4.00, 3.00, 2.00, 1.00, 0.  , 1.00, 2.00],
                       [9.00, 8.00, 7.00, 6.00, 5.00, 4.00, 3.00, 2.00, 1.00, 0.  , 1.00],
                       [10.0, 9.00, 8.00, 7.00, 6.00, 5.00, 4.00, 3.00, 2.00, 1.00, 0.  ]]),
        2)
        self.assertTrue(np.allclose(g1, should_be, rtol=1e-10, atol=1e-20))
        self.assertTrue(np.allclose(g1n, should_be/should_be.max(), rtol=1e-10, atol=1e-20))
        
        # Check with randomly generated samples
        tcs = tutils.build_tcs(
            ["description", "samples"],
            [            
                ["Small randomly generated sample", 
                    tutils.generate_random_sample(2, 2)],
                ["Medium randomly generated sample", 
                    tutils.generate_random_sample(10, 2)],
                ["Big randomly generated sample", 
                    tutils.generate_random_sample(100, 2)]
            ])
        for i, tc in enumerate(tcs):
            try:
                gram, nvalue = kernel.get_squared_gram(tc.samples, normalize=False)
                sdmax = 0
                for i in range(gram.shape[0]):
                    for j in range(gram.shape[1]):
                        sdiff = df(tc.samples[i, :], tc.samples[j, :]) ** 2
                        if sdmax < sdiff:
                            sdmax = sdiff
                        self.assertTrue(abs((gram[i, j] - sdiff)) < 1e-20)
                    
                gram, nvalue = kernel.get_squared_gram(tc.samples, normalize=True)
                for i in range(gram.shape[0]):
                    for j in range(gram.shape[1]):
                        self.assertTrue(abs((gram[i, j] - df(tc.samples[i, :], tc.samples[j, :])**2/sdmax)) < 1e-20)
            except AssertionError, e:
                print "\n[TEST] Assert violated in test case %d: %s\n" % (i, tc.description)
                raise e
    
    def test_gram_to_kernel(self):
        df = lambda x_1, x_2: np.linalg.norm(x_1 - x_2)
        sigma = 2.
        lengthscale = 1.
        noise = 0.001
        params = [sigma, lengthscale, noise, df]
        kernel = RBF(*params)
        
        calculate_rbf = lambda gram: (sigma * np.exp((-0.5/lengthscale) * gram)) + np.eye(gram.shape[0]) * noise
        
        constants = np.matrix(np.linspace(-5, 5, 11)).transpose()
        rsmall = tutils.generate_random_sample(2, 2)
        rmid = tutils.generate_random_sample(10, 2)
        rbig = tutils.generate_random_sample(100, 2)
        tcs = tutils.build_tcs(
            ["description", "samples", "kernel"],
            [            
                ["Constant sample", 
                    constants, calculate_rbf(kernel.get_squared_gram(constants, normalize=False)[0])],
                ["Small randomly generated sample", 
                    rsmall, calculate_rbf(kernel.get_squared_gram(rsmall, normalize=False)[0])],
                ["Medium randomly generated sample", 
                    rmid, calculate_rbf(kernel.get_squared_gram(rmid, normalize=False)[0])],
                ["Big randomly generated sample", 
                    rbig, calculate_rbf(kernel.get_squared_gram(rbig, normalize=False)[0])]
            ])
        for i, tc in enumerate(tcs):
            try:
                gram, nvalue = kernel.get_squared_gram(tc.samples, normalize=False)
                tutils.assertEqualMatrix(self, kernel.gram_to_kernel(gram), tc.kernel)
            except AssertionError, e:
                print "\n[TEST] Assert violated in test case %d: %s\n" % (i, tc.description)
                raise e
        
    def test_call_equals_gram_to_kernel(self):
        # The normal call of the RBF and gram2kernel(gram(data)) should be the same.
        df = lambda x_1, x_2: np.linalg.norm(x_1 - x_2)
        sigma = 2.
        lengthscale = 1.
        noise = 0.001
        params = [sigma, lengthscale, noise, df]
        kernel = RBF(*params)
        
        calculate_rbf = lambda gram: (sigma * np.exp((-0.5/lengthscale) * gram)) + np.eye(gram.shape[0]) * noise
        
        tcs = tutils.build_tcs(
            ["description", "samples"],
            [            
                ["Constant sample",
                    np.matrix(np.linspace(-5, 5, 11)).transpose()],
                ["Small randomly generated sample",
                    tutils.generate_random_sample(2, 2)],
                ["Medium randomly generated sample",
                    tutils.generate_random_sample(10, 2)],
                ["Big randomly generated sample",
                    tutils.generate_random_sample(100, 2)]
            ]
        )
        for t, tc in enumerate(tcs):
            try:
                gram, nvalue = kernel.get_squared_gram(tc.samples, normalize=False)
                K = kernel.gram_to_kernel(gram)
                for i in range(K.shape[0]):
                    for j in range(K.shape[1]):
                        k_value = K[i, j]
                        if i == j:
                            k_value -= noise
                        self.assertEqual(k_value, kernel(tc.samples[i, :], tc.samples[j, :]))
                
                gram, nvalue = kernel.get_squared_gram(tc.samples, normalize=True)
                K = kernel.gram_to_kernel(gram)
                for i in range(K.shape[0]):
                    for j in range(K.shape[1]):
                        k_value = K[i, j]
                        if i == j:
                            k_value -= noise
                        self.assertEqual(k_value, kernel(tc.samples[i, :], tc.samples[j, :], normalize_by=nvalue))
            except AssertionError, e:
                print "\n[TEST] Assert violated in test case %d: %s\n" % (i, tc.description)
                raise e

def build_model(dims, optimize_noise=False, normalize_gram=False, kernel_params=[2., 1., 0.001]):
    df = lambda x_1, x_2: np.linalg.norm(x_1 - x_2)
    params = kernel_params + [df] # lengthscale, mult, noise, df
    kernel = RBF(*params)
    model = GaussianProcessModel(kernel, dims[0], dims[1], optimize_noise=optimize_noise, normalize_gram=normalize_gram)
    return kernel, model
    
def train_skl_model(samples, observations):
    skl_lengthscale, skl_lengthscale_bounds = 10., (1e-2, 1e2)
    skl_mult, skl_mult_bounds = 1.0, (1e-3, 1e3)
    skl_kf = C(skl_mult, skl_mult_bounds) * SKRBF(skl_lengthscale, skl_lengthscale_bounds)
    skl_gp = GaussianProcessRegressor(kernel=skl_kf, normalize_y=True, n_restarts_optimizer=10)
    skl_gp.fit(samples, observations.transpose())
    return skl_gp

class TestGaussianProcessModel(unittest.TestCase):
    
    def test_init(self):
        # Test the kernels, dims and options are set correctly
        tcs = tutils.build_tcs(
            ["description", "dims", "optimize_noise", "normalize_gram"],
            [            
                ["1D fun no-noise no-gram", 
                    [1, 1], False, False],
                ["2D fun no-noise gram", 
                    [2, 1], False, True],
                ["2D2I fun noise no-gram", 
                    [2, 2], True, False],
                ["10D2I fun noise gram", 
                    [10, 2], True, True]
            ])
        for i, tc in enumerate(tcs):
            try:
                kernel, model = build_model(tc.dims, tc.optimize_noise, tc.normalize_gram)
                self.assertEqual(model.kf, kernel)
                self.assertEqual([model.dim_in, model.dim_out], tc.dims)
                self.assertEqual(model.optimize_noise, tc.optimize_noise)
                self.assertEqual(model.normalize_gram, tc.normalize_gram)
            except AssertionError, e:
                print "\n[TEST] Assert violated in test case %d: %s\n" % (i, tc.description)
                raise e
        
    def test_data_assignment_obs_mean(self):
        # Test compute with compute_kernel = False
        def f(x, y):
            return np.sin(np.sqrt(x ** 2 + y ** 2))
        kernel, model = build_model([2, 1], optimize_noise=False, normalize_gram=False)
        samples, observations = tutils.generate_random_sample(20, 2, func=f)
        
        # Test with matrix of correct dimensions
        model.compute(samples, observations.transpose(), compute_kernel=False)
        tutils.assertEqualMatrix(self, model.samples, samples)
        tutils.assertEqualMatrix(self, model.observations, observations.transpose())
        self.assertEqual(model.observations_mean, np.mean(observations))
        tutils.assertEqualMatrix(self, model.zero_meaned_obs, observations.transpose() - np.mean(observations.transpose()))
        
        # Test with array (should cast and transpose automatically)
        model.compute(samples, tutils.to_array(observations), compute_kernel=False)
        tutils.assertEqualMatrix(self, model.observations, observations.transpose())
        self.assertEqual(model.observations_mean, np.mean(observations))
        tutils.assertEqualMatrix(self, model.zero_meaned_obs, observations.transpose() - np.mean(observations.transpose()))
    
    def test_compute(self):
        def f(x, y):
            return np.sin(np.sqrt(x ** 2 + y ** 2))
        
        constants = tutils.to_array(np.matrix(np.linspace(-5, 5, 11)).transpose())
        tcs = tutils.build_tcs(
            ["description", "samples", "observations", "dims", "normalized"],
            [            
                ["Constant sample unnormalized", 
                    constants, np.matrix(f(constants, 0)), (1, 1), False],
                ["Small randomly generated sample unnormalized"] + 
                    list(tutils.generate_random_sample(2, 2, func=f)) + [(2, 1), False],
                ["Medium randomly generated sample unnormalized"] + 
                    list(tutils.generate_random_sample(10, 2, func=f)) + [(2, 1), False],
                ["Big randomly generated sample unnormalized"] + 
                    list(tutils.generate_random_sample(100, 2, func=f)) + [(2, 1), False],
                ["Constant sample normalized", 
                    constants, np.matrix(f(constants, 0)), (1, 1), True],
                ["Small randomly generated sample normalized"] + 
                    list(tutils.generate_random_sample(2, 2, func=f)) + [(2, 1), True],
                ["Medium randomly generated sample normalized"] + 
                    list(tutils.generate_random_sample(10, 2, func=f)) + [(2, 1), True],
                ["Big randomly generated sample normalized"] + 
                    list(tutils.generate_random_sample(100, 2, func=f)) + [(2, 1), True]
            ])
        for i, tc in enumerate(tcs):
            try:
                # Build the model and test assignation.
                kernel, model = build_model(tc.dims, optimize_noise=False, normalize_gram=tc.normalized)
                model.compute(tc.samples, tc.observations.transpose(), compute_kernel=True)
                tutils.assertEqualMatrix(self, model.samples, tc.samples)
                tutils.assertEqualMatrix(self, model.observations, tc.observations.transpose())
                self.assertEqual(model.observations_mean, np.mean(tc.observations))
                zero_meaned_obs = tc.observations.transpose() - np.mean(tc.observations.transpose())
                tutils.assertEqualMatrix(self, model.zero_meaned_obs, zero_meaned_obs)
                
                # Test the gram matrix is set and correct.
                sgram, nvalue = kernel.get_squared_gram(tc.samples, normalize=tc.normalized)
                if tc.normalized:
                    usgram, _ = kernel.get_squared_gram(tc.samples, normalize=False)
                    self.assertEqual(nvalue, usgram.max())
                    self.assertEqual(model.gram_nvalue, usgram.max())
                else:
                    self.assertEqual(nvalue, 1.)
                    self.assertEqual(model.gram_nvalue, 1.)
                tutils.assertEqualMatrix(self, model.squared_gram, sgram)
                
                # Test the kernel matrix is set and correct.
                K = kernel.gram_to_kernel(sgram)
                tutils.assertEqualMatrix(self, model.kernel, K)
                
                # Test matrixL is lower and the cholesky factorization is correct.
                upper_mL = np.triu(model.matrixL)
                np.fill_diagonal(upper_mL, 0.)
                tutils.assertEqualMatrix(self, upper_mL, np.zeros(K.shape))
                tutils.assertEqualMatrix(self, np.dot(model.matrixL, model.matrixL.transpose()), K) # K = LL* (Cholesky)
                
                # Test alpha contains the inverse of the kernel multiplied by the value of the observations.
                K_inv = np.linalg.inv(K)
                tutils.assertEqualMatrix(self, model.alpha, np.dot(K_inv, zero_meaned_obs))
                
                # Test K_inv is not set (for performance reasons, while optimizing we
                # dont need the inverse to calculate the Likelihood).
                self.assertEqual(model.K_inv.size, 0)
            except AssertionError, e:
                print "\n[TEST] Assert violated in test case %d: %s\n" % (i, tc.description)
                raise e
    
    def test_recompute(self):
        # Recompute performs a subset of operations of compute by leaving out 
        # the assignation of samples and observations, and the computation of
        # the gram kernel.
        def f(x, y):
            return np.sin(np.sqrt(x ** 2 + y ** 2))
        
        constants = tutils.to_array(np.matrix(np.linspace(-5, 5, 11)).transpose())
        tcs = tutils.build_tcs(
            ["description", "samples", "observations", "dims", "normalized"],
            [            
                ["Constant sample unnormalized", 
                    constants, np.matrix(f(constants, 0)), (1, 1), False],
                ["Small randomly generated sample unnormalized"] + 
                    list(tutils.generate_random_sample(2, 2, func=f)) + [(2, 1), False],
                ["Medium randomly generated sample unnormalized"] + 
                    list(tutils.generate_random_sample(10, 2, func=f)) + [(2, 1), False],
                ["Big randomly generated sample unnormalized"] + 
                    list(tutils.generate_random_sample(100, 2, func=f)) + [(2, 1), False],
                ["Constant sample normalized", 
                    constants, np.matrix(f(constants, 0)), (1, 1), True],
                ["Small randomly generated sample normalized"] + 
                    list(tutils.generate_random_sample(2, 2, func=f)) + [(2, 1), True],
                ["Medium randomly generated sample normalized"] + 
                    list(tutils.generate_random_sample(10, 2, func=f)) + [(2, 1), True],
                ["Big randomly generated sample normalized"] + 
                    list(tutils.generate_random_sample(100, 2, func=f)) + [(2, 1), True]
            ])
        for i, tc in enumerate(tcs):
            try:
                # Build the model and test assignation.
                kernel, model = build_model(tc.dims, optimize_noise=False, normalize_gram=tc.normalized)
                model.compute(tc.samples, tc.observations.transpose(), compute_kernel=True)
                
                # Change the hiperparameters and recompute to check that 
                # everything is recalculated correctly.
                new_params = map(lambda p: p*2, model.kf.get_hps())
                kernel.set_hps(*new_params)
                model.kf.set_hps(*new_params)
                
                # Recompute the model with the new hiperparameters.
                model.recompute()
                
                # Calculate the zero meaned observations to use for comparison.
                zero_meaned_obs = tc.observations.transpose() - \
                    np.mean(tc.observations.transpose())
                
                # Test the gram matrix is set and correct.
                sgram, nvalue = kernel.get_squared_gram(tc.samples, normalize=tc.normalized)
                if tc.normalized:
                    usgram, _ = kernel.get_squared_gram(tc.samples, normalize=False)
                    self.assertEqual(nvalue, usgram.max())
                    self.assertEqual(model.gram_nvalue, usgram.max())
                else:
                    self.assertEqual(nvalue, 1.)
                    self.assertEqual(model.gram_nvalue, 1.)
                tutils.assertEqualMatrix(self, model.squared_gram, sgram)
                
                # Test the kernel matrix is set and correct.
                K = kernel.gram_to_kernel(sgram)
                tutils.assertEqualMatrix(self, model.kernel, K)
                
                # Test matrixL is lower and the cholesky factorization is correct.
                upper_mL = np.triu(model.matrixL)
                np.fill_diagonal(upper_mL, 0.)
                tutils.assertEqualMatrix(self, upper_mL, np.zeros(K.shape))
                tutils.assertEqualMatrix(self, np.dot(model.matrixL, model.matrixL.transpose()), K) # K = LL* (Cholesky)
                
                # Test alpha contains the inverse of the kernel multiplied by the value of the observations.
                K_inv = np.linalg.inv(K)
                tutils.assertEqualMatrix(self, model.alpha, np.dot(K_inv, zero_meaned_obs))
                
                # Test K_inv is not set (for performance reasons, while optimizing we
                # dont need the inverse to calculate the Likelihood).
                self.assertEqual(model.K_inv.size, 0)
            except AssertionError, e:
                print "\n[TEST] Assert violated in test case %d: %s\n" % (i, tc.description)
                raise e
    
    def test_predict(self):
        def f(x, y):
            return np.sin(np.sqrt(x ** 2 + y ** 2))
        
        np.random.seed(0)
        constants = tutils.to_array(np.matrix(np.linspace(-5, 5, 11)).transpose())
        tcs = tutils.build_tcs(
            ["description", "samples", "observations", "to_predict", "dims", "normalized"],
            [            
                ["Constant sample unnormalized", 
                    constants, np.matrix(f(constants, 0)), \
                    [0.25], (1, 1), False],
                ["Small randomly generated sample unnormalized"] + 
                    list(tutils.generate_random_sample(2, 2, func=f)) + \
                    [tutils.generate_random_sample(10, 2), (2, 1), False],
                ["Medium randomly generated sample unnormalized"] + 
                    list(tutils.generate_random_sample(10, 2, func=f)) + \
                    [tutils.generate_random_sample(10, 2), (2, 1), False],
                ["Big randomly generated sample unnormalized"] + 
                    list(tutils.generate_random_sample(100, 2, func=f)) + \
                    [tutils.generate_random_sample(10, 2), (2, 1), False],
                ["Constant sample normalized", 
                    constants, np.matrix(f(constants, 0)), \
                    [0.25], (1, 1), True],
                ["Small randomly generated sample normalized"] + 
                    list(tutils.generate_random_sample(2, 2, func=f)) + \
                    [tutils.generate_random_sample(10, 2), (2, 1), True],
                ["Medium randomly generated sample normalized"] + 
                    list(tutils.generate_random_sample(10, 2, func=f)) + \
                    [tutils.generate_random_sample(10, 2), (2, 1), True],
                ["Big randomly generated sample normalized"] + 
                    list(tutils.generate_random_sample(100, 2, func=f)) + \
                    [tutils.generate_random_sample(10, 2), (2, 1), True],
            ])
        for t, tc in enumerate(tcs):
            try:
                # Build the model.
                kernel, model = build_model(tc.dims, optimize_noise=False, normalize_gram=tc.normalized)
                model.compute(tc.samples, tc.observations.transpose(), compute_kernel=True)
                
                # Calculate everything needed for comparison.
                observations_mean = np.mean(tc.observations.transpose())
                zero_meaned_obs = tc.observations.transpose() - observations_mean
                sgram, nvalue = kernel.get_squared_gram(tc.samples, normalize=tc.normalized)
                K = kernel.gram_to_kernel(sgram)
                K_inv = np.linalg.inv(K)
                alpha = np.dot(K_inv, zero_meaned_obs)
                
                # Choose a new point from the domain to predict, calculate the 
                # row of the kernel, and predict mean and variance.
                m = tc.samples.shape[0]
                for x in tc.to_predict:
                    k = np.zeros((m, 1))
                    for i in range(m):
                        k[i] = kernel(tc.samples[i], x, normalize_by=nvalue)
                    mean = tutils.to_array(np.dot(k.transpose(), alpha) + observations_mean)
                    variance = (kernel(x, x, normalize_by=nvalue) - np.dot(k.transpose(), K_inv).dot(k))[0]
                    deviation = np.sqrt(variance)
                    if deviation.size == 1:
                        deviation = deviation[0]
                    if mean.size == 1:
                        mean = mean[0]
                    pmean, pdeviation = model.predict(x)
                    npt.assert_array_almost_equal([pmean, pdeviation], [mean, deviation], decimal=8) # this is the limit of decimals that can be trusted
                
                if not tc.normalized:
                    # Compare against SKLearn too.
                    if tc.dims[0] == 1:
                        skl_gp = train_skl_model(tc.samples.reshape(-1, 1), tc.observations)
                    else:
                        skl_gp = train_skl_model(tc.samples, tc.observations)
                    skl_hparams = np.exp(skl_gp.kernel_.theta)
                    model.kf.set_hps(*([skl_hparams[0], skl_hparams[1]**2, 0.0]))
                    model.compute(tc.samples, tc.observations.transpose(), compute_kernel=True)
                    
                    # Compute the points to compare
                    mean, deviation = model.predict(np.array(tc.to_predict))
                    if tc.dims[0] == 1:
                        skl_mean, skl_deviation = skl_gp.predict(np.array(tc.to_predict).reshape(-1, 1), return_std=True)
                        skl_mean, skl_deviation = skl_mean[0], skl_deviation[0]
                    else:
                        skl_mean, skl_deviation = skl_gp.predict(np.array(tc.to_predict), return_std=True)
                        skl_mean = skl_mean[:, 0]
                    npt.assert_array_almost_equal([mean, deviation], [skl_mean, skl_deviation], decimal=4)
                
                # Test K_inv is set now, since we did prediction.
                self.assertNotEqual(model.K_inv.size, 0)
            except AssertionError, e:
                print "\n[TEST] Assert violated in test case %d: %s\n" % (t, tc.description)
                raise e
    
    def test_get_log_likelihood_by_calculation(self):
        def f(x):
            return 0.75 * (2.5+x) * np.sin(2.5+x)
        
        for i in range(10):
            np.random.seed(i)
            in_dims, out_dims = (1, 1)
            samples, observations = tutils.generate_random_sample(10, in_dims, func=f, bounds=[-10, 10])
            
            kernel, gp = build_model((in_dims, out_dims), optimize_noise=False, normalize_gram=False)
            gp.compute(samples, observations.transpose(), compute_kernel=True)
            
            # Obtained from sklearn.GaussianProcessRegresor
            log_likelihood_dims = -0.5 * np.einsum("ik,ik->k", gp.zero_meaned_obs, gp.alpha)
            log_likelihood_dims -= np.log(np.diag(gp.matrixL)).sum()
            log_likelihood_dims -= gp.kernel.shape[0] / 2 * np.log(2 * np.pi)
            log_likelihood = log_likelihood_dims.sum(-1)  # sum over dimensions
            
            self.assertTrue(abs(log_likelihood - gp.get_log_likelihood()) < 1e-5)
    
    def test_get_log_likelihood_by_comparision(self):
        # Verify the computation is made correctly, we take sklearn to help us 
        # in this task.
        def f(x):
            return 0.75 * (2.5+x) * np.sin(2.5+x)
        
        # We test with 100 different sets of samples and observations. We set
        # the seed to avoid a flaky test.
        reps = 10
        avg_err = 0
        np.random.seed(0)
        for i in range(reps):
            in_dims, out_dims = (1, 1)
            samples, observations = tutils.generate_random_sample(10, in_dims, func=f, bounds=[-10, 10])
            
            # Restore random seed
            np.random.seed(None)
            
            # Instantiate and train the SKL GP.
            skl_gp = train_skl_model(samples, observations)
            
            # NOTE: Why SKL is squaring the parameters?
            skl_hparams = np.exp(skl_gp.kernel_.theta) # sigma, lengthscale
            hparams = [skl_hparams[0], skl_hparams[1]**2., 0.0] # sigma, lengthscale, noise
            kernel, gp = build_model((in_dims, out_dims), optimize_noise=False, normalize_gram=False, kernel_params=hparams)
            gp.compute(samples, observations.transpose(), compute_kernel=True)
            
            # Test the likelihood calculation
            error = abs(gp.get_log_likelihood() - skl_gp.log_marginal_likelihood())
            avg_err += error
        
        avg_error = float(avg_err)/reps
        self.assertTrue(avg_error < 1e-2)
    
    def test_optimize(self):
        # TODO: Test the optimization is finding good parameters for the data (a 
        # comparison with before/after could be made to check this).
        # Test the PD errors is zero when using the data from a function.
        # -- obs: maybe the PD errors are caused because of rounding problems?
        # Test the parameters that CMA-ES finds are the ones being set.
        # Test the recomputation of the kernel with the new paramaters takes 
        # place after the optimization is finished.
        pass
    
    def test_gp_with_1d_function(self):
        # Test the GP estimation against a 1d function.
        do_plots = False
        for i in range(10):
            np.random.seed(i)
            def f(x):
                return 0.75 * (2.5+x) * np.sin(2.5+x)
            
            # Define dimensions and initialize, samples, observations and
            # points to interpolate.
            in_dims, out_dims = (1, 1)
            samples, observations = tutils.generate_random_sample(10, in_dims, func=f, bounds=[-10, 10])
            t = np.linspace(-10, 10, 200)
            
            # Restore random seed
            np.random.seed(None)
            
            # Instantiate and train the SKL GP.
            skl_gp = train_skl_model(samples, observations)
            
            # TODO: Why SKL is squaring the lengthscale?
            skl_hparams = np.exp(skl_gp.kernel_.theta) # sigma, lengthscale
            
            # Instantiate and train this GP.
            hparams = [skl_hparams[0], skl_hparams[1]**2., 0.0] # sigma, lengthscale, noise
            kernel, gp = build_model((in_dims, out_dims), optimize_noise=False, normalize_gram=False, kernel_params=hparams)
            gp.compute(samples, observations.transpose(), compute_kernel=True)
            gp.optimize(bounded=True, fevals=10000, pub=[1e4, 1e3, 0.0001], plb=[1e-5, 1e-5, 1e-7])
            # print [skl_hparams[0], skl_hparams[1]**2.]
            
            # Predict the testing samples
            means, deviations = gp.predict(t)
            skl_means, skl_deviations = skl_gp.predict(np.matrix(t).transpose(), return_std=True)
            skl_means = skl_means[:, 0]
            
            if abs(gp.get_log_likelihood() - \
                skl_gp.log_marginal_likelihood()) > 1e-2:
                # print gp.hp_not_positive_definite_error_count
                print gp.get_log_likelihood()
                print skl_gp.log_marginal_likelihood()
                continue
            
            if do_plots:
                fig, ax = plt.subplots()
                ax.plot(t, f(t), label="x sin(x)")

                gp_plot = ax.plot(t, means, label="GP median")
                ax.fill_between(t, means - deviations, means + deviations, alpha=0.5)

                sklgp_plot = ax.plot(t, skl_means, label="SKL-GP median")
                ax.fill_between(t, skl_means - skl_deviations, skl_means + skl_deviations, alpha=0.5)

                ax.errorbar(samples, observations, fmt='ok', zorder=3)
                ax.legend()

                plt.ylim((-10, 10))
                plt.savefig(str(i)+"_output.png")
                # plt.show()
            
            npt.assert_array_almost_equal([means, deviations], [skl_means, skl_deviations], decimal=2)
    
    def test_stability(self):
        # TODO: Test the GP estimation with low lengthscale comparing it to the
        # implementation of sklearn.
        pass

if __name__ == '__main__':
    unittest.main()