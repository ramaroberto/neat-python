"""Implements surrogate assitance model"""

from __future__ import print_function
from math import sqrt

from neat.config import ConfigParameter, DefaultClassConfig
from neat.models.gp import RBF, GaussianProcessModel

import time
import numpy as np
import dill
import pickle
import abc

class DefaultSurrogateModel(object):
    
    @classmethod
    def parse_config(cls, param_dict):
        return DefaultClassConfig(param_dict,
                                  [ConfigParameter('enabled', bool, False),
                                  ConfigParameter('number_initial_samples', int, 32),
                                  ConfigParameter('gens_per_infill', int, 4),
                                  ConfigParameter('number_infill_individuals', int, 4),
                                  ConfigParameter('resolve_threshold', int, 128),
                                  ConfigParameter('max_training_set_size', int, 128),
                                  ConfigParameter('distance_function', str, 'neat'),
                                  ConfigParameter('surrogate_model', str, 'gp')])
    
    def __init__(self, config, reporters):
        # pylint: disable=super-init-not-called
        self.surrogate_config = config.surrogate_config
        self.reporters = reporters
        
        self.model = None
        self.fitness_function = None
        self.old_training_set = []
        self.training_set = []
        self.training_set_ids = set([])
        self.max_training_set_size = self.surrogate_config.max_training_set_size
        
        self.distance_similarity_threshold = 0.1
        
        self.distance_functions_map = {
            'neat': lambda g1, g2: g1.distance(g2, config.genome_config)
        }
        self.distance_function = self.distance_functions_map[self.surrogate_config.distance_function]
        
        self.surrogate_models_map = {
            'gp': GaussianProcessSurrogateModel,
            'fake': FakeSurrogateModel
        }
        self.surrogate_model_params_map = {
            'gp': {
                'distance_function': self.distance_function
            },
            'fake': None
        }
        self.surrogate_model_class = self.surrogate_models_map[self.surrogate_config.surrogate_model]
        self.surrogate_model_params = self.surrogate_model_params_map[self.surrogate_config.surrogate_model]
    
    def samples_length(self):
        return len(self.old_training_set) + len(self.training_set)
    
    def reset(self):
        self.model = None
        self.old_training_set = self.training_set
        self.training_set = []
        
    def get_from_training(self, n):
        n = len(self.training_set) - min(n, len(self.training_set))
        return self.training_set[n:]
    
    def add_to_training(self, genomes):
        """Add all the genomes to the training set."""
        # Append the new genomes to the back of the training set.
        self.training_set += filter(lambda g: g.key not in self.training_set_ids, genomes)
        self.training_set_ids = self.training_set_ids.union(set(map(lambda g: g.key, genomes)))
        
        training_set_size = len(self.training_set) + len(self.old_training_set)
        to_remove = training_set_size - self.max_training_set_size
        if 0 < to_remove:
            removed = []
            if len(self.old_training_set) < to_remove:
                to_remove = max(to_remove - len(self.old_training_set), 0)
                removed = self.old_training_set + self.training_set[:to_remove]
                self.old_training_set = []
                self.training_set = self.training_set[to_remove:]
            else:
                removed = self.old_training_set[:to_remove]
                self.old_training_set = self.old_training_set[to_remove:]
        
            # Delete the removed genomes from the set of ids.
            for genome in removed:
                self.training_set_ids.remove(genome.key)
        
        assert len(self.training_set) + len(self.old_training_set) <= self.max_training_set_size
        
    def update_training(self, species, fitness_function, config):
        """Takes the genomes in the species and updates the training set."""
        # Precondition: Species should have the real fitness set.
    
        # Order the species by descending fitness.
        ranked_species = sorted(species.values(), key=lambda s: s.fitness, reverse=True)
        
        to_infill = self.surrogate_config.number_infill_individuals
        all_genomes = []
        best_genomes = []
        
        # First we add the best genome of each species.
        for s in ranked_species:
            ranked_genomes = sorted(s.members.values(), key=lambda g: -g.fitness)
            for i, g in enumerate(ranked_genomes):
                if g.key not in self.training_set_ids:
                    best_genomes.append(g)
                    all_genomes += ranked_genomes[i+1:] # TODO: could be improved with merge
                    to_infill -= 1
                    break
            if to_infill <= 0: # More species than genomes needed to infill
                break
        
        # Then we fill the remaining spots.
        all_genomes.sort(key=lambda g: -g.fitness)
        while 0 < to_infill:
            genome = all_genomes.pop(0)
            if genome.key in self.training_set_ids:
                continue
            best_genomes.append(genome)
            to_infill -= 1
        
        # Evaluate with the real fitness function and add to training.
        best_genomes.sort(key=lambda g: g.fitness, reverse=True)
        print("[SURR]", len(best_genomes), map(lambda g: (g.key, g.fitness), best_genomes))
        fitness_function(map(lambda g: (g.key, g), best_genomes), config)
        print("[SURR]", len(best_genomes), map(lambda g: (g.key, g.fitness), best_genomes))

        self.add_to_training(best_genomes)
        return best_genomes
    
    def is_training_set_new(self):
        return len(self.old_training_set) == 0
        
    def evaluate(self, population, generation, fitness_function, config, testing=False): # NOTE: Generation and fitness_function only required for testing
        genomes = filter(lambda g: not g.real_fitness, population.values())
        predictions = self.model.predict(genomes, fitness_function)
        for i, genome in enumerate(genomes):
            genome.fitness = predictions[i]
    
    def train(self, generation, config, optimize=False): # NOTE: Generation only required for testing
        if self.surrogate_config.surrogate_model == 'fake' \
            and not self.surrogate_model_params:
            self.surrogate_model_params = {
                'config': config,
            }
        self.training_set = self.old_training_set + self.training_set
        self.old_training_set = []
        
        if self.model is None:
            self.model = self.surrogate_model_class.initialize(self.surrogate_model_params)
            optimize = True
        self.model.train(self.training_set, map(lambda g: g.real_fitness, self.training_set), optimize=optimize)
        
        # NOTE: For testing only
        # with open("data/"+str(generation)+"_train_genomes.pkl", "wb") as output:
        #     pickle.dump(self.training_set, output, pickle.HIGHEST_PROTOCOL)
        # self.model.compute(self.training_set, map(lambda g: g.real_fitness, self.training_set), compute_kernel=False)
        # with open("data/"+str(generation)+"_model.pkl", "wb") as output:
        #     pickle.dump(self.model, output, pickle.HIGHEST_PROTOCOL)
        # print(self.model.kernel[0])

# initialize(df), train(samples, observations), predict(samples)

class SurrogateModel(object):
    __metaclass__ = abc.ABCMeta
    
    @classmethod
    @abc.abstractmethod
    def initialize(self, params):
        """Initialize the surrogate model."""
        return
    
    @abc.abstractmethod
    def train(self, samples, observations, optimize=False):
        """Train the model with the collected samples and observations."""
        return
    
    @abc.abstractmethod
    def predict(self, samples, fitness_function):
        """Predict samples' fitness using the trained model."""
        return
        
class GaussianProcessSurrogateModel(object):
    @classmethod
    def initialize(self, params):
        # TODO: Error if no distance function defined in params dict.
        return self(params['distance_function'])
    
    def __init__(self, distance_function):
        # self.acquisition = lambda mu, std: mu + std
        ucb_coef = 1e-3
        self.acquisition = lambda mu, std: mu + ucb_coef * std
        # self.acquisition = lambda mu, std: mu
        self.kernel = RBF(sigma=0.001, length=5, noise=1e-3, df=distance_function)
        self.model = GaussianProcessModel(self.kernel, 1, 1, \
            optimize_noise=True, normalize_gram=False)
        self.filter_nearby = True

    def train(self, samples, observations, optimize=False):
        if self.filter_nearby:
            filtered_samples = []
            filtered_observations = []
            for g, obs in zip(samples, observations):
                too_close = False
                for i in range(len(filtered_samples)):
                    if self.model.kf.df(g, filtered_samples[i]) < 1e-1:
                        too_close = True
                        if g.real_fitness < filtered_samples[i].real_fitness:
                            filtered_samples[i] = g
                            filtered_observations[i] = g.real_fitness
                        break
                if not too_close:
                    filtered_samples.append(g)
                    filtered_observations.append(obs)
            samples = filtered_samples
            observations = filtered_observations
        self.model.compute(samples, observations)
        if optimize:
            self.model.optimize(quiet=True, bounded=True, fevals=200)

    def predict(self, samples, fitness_function):
        """Predict samples fitness using the trained model."""
        predictions = []
        for sample in samples:
            mu, std = self.model.predict(sample)
            predictions.append(self.acquisition(mu, std))
        return predictions
            
class FakeSurrogateModel(object):
    @classmethod
    def initialize(self, params):
        # TODO: Error if no distance function defined in params dict.
        return self(params['config'])
    
    def __init__(self, config):
        # self.fitness_function = config.fitness_function
        self.config = config

    def train(self, samples, observations):
        print("Fake Training... (just sleeping actually... LOL, YOLO)")
        time.sleep(3)
        pass

    def predict(self, samples, fitness_function):
        """Predict samples fitness using the trained model."""
        predictions = []
        for sample in samples:
            fitness_function([(1, sample)], self.config)
            fitness = sample.fitness
            sample.real_fitness = None
            noisy_fitness = sample.fitness + \
                abs(np.random.normal(0., fitness/4.))
            # normal noise with 1/4 variance of fitness and UCB emulation.
            predictions.append(noisy_fitness)
        return predictions