"""Implements surrogate assitance model"""

from __future__ import print_function
from math import sqrt

from neat.config import ConfigParameter, DefaultClassConfig
from neat.models.gp import RBF, GaussianProcessModel

import dill
import pickle

class DefaultSurrogateModel(object):
    
    @classmethod
    def parse_config(cls, param_dict):
        return DefaultClassConfig(param_dict,
                                  [ConfigParameter('enabled', bool, False),
                                  ConfigParameter('number_initial_samples', int, 32),
                                  ConfigParameter('gens_per_infill', int, 4),
                                  ConfigParameter('number_infill_individuals', int, 4),
                                  ConfigParameter('resolve_threshold', int, 128),
                                  ConfigParameter('max_training_set_size', int, 128)])
    
    def __init__(self, config, reporters):
        # pylint: disable=super-init-not-called
        self.surrogate_config = config
        self.reporters = reporters
        
        self.model = None
        self.old_training_set = []
        self.training_set = []
        self.training_set_ids = set([])
        self.max_training_set_size = self.surrogate_config.max_training_set_size
        
        # UCB
        self.acquisition = lambda mu, std: mu + std
        
        self.distance_similarity_check = True
        self.distance_similarity_threshold = 0.1
    
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
        # Precondition: Species should have the fitness set.
    
        # Order the species by descending fitness.
        ranked_species = sorted(map(lambda s: (s.fitness if s.fitness else 0.0, s), species.values()), key=lambda s: -s[0])
        
        to_infill = self.surrogate_config.number_infill_individuals
        all_genomes = []
        best_genomes = []
        
        # First we add the best genome of each species.
        for f, s in ranked_species:
            ranked_genomes = sorted(s.members.values(), key=lambda g: -g.fitness)
            limit = min(len(ranked_genomes), to_infill)
            all_genomes += ranked_genomes[1:] # TODO: could be improved with merge
            if ranked_genomes[0].key in self.training_set_ids:
                continue
            best_genomes.append(ranked_genomes[0])
            
            to_infill -= 1
            if to_infill <= 0: # More species than genomes needed to infill
                break
        
        
        # Then we fill the remaining spots.
        all_genomes.sort(key=lambda g: -g.fitness)
        while 0 < to_infill:
            genome = all_genomes.pop(0)
            if genome.key in self.training_set_ids:
                continue
            
            if self.distance_similarity_check:
                # Avoid having too similar genomes in the training samples.
                # NOTE: Otherwise, this could cause numerical problems.
                if self._is_similar(genome, config, additional_genomes=best_genomes):
                    continue
            
            best_genomes.append(genome)
            to_infill -= 1
        
        # Evaluate with the real fitness function and add to training.
        print("[SURR]", len(best_genomes), map(lambda g: (g.key, g.fitness), best_genomes))
        fitness_function(map(lambda g: (g.key, g), best_genomes), config)
        print("[SURR]", len(best_genomes), map(lambda g: (g.key, g.fitness), best_genomes))

        self.add_to_training(best_genomes)
        return best_genomes
    
    def _is_similar(self, genome, config, additional_genomes=[]):
        df = lambda g1, g2: g1.distance(g2, config.genome_config)
        for bg in additional_genomes+self.old_training_set+self.training_set:
            if df(bg, genome) < self.distance_similarity_threshold:
                return True
        return False
    
    def is_training_set_new(self):
        return len(self.old_training_set) == 0
        
    def evaluate(self, population, generation, fitness_function, config, testing=False): # NOTE: Generation and fitness_function only required for testing
        evaluated_genomes = []
        for genome in population.values():
            if not genome.real_fitness:
                if testing:
                    fitness_function(map(lambda g: (g.key, g), [genome]), config)
                    evaluated_genomes.append(genome)
                    with open("data/"+str(generation)+"_test_genomes.pkl", "wb") as output:
                        pickle.dump(self.training_set, output, pickle.HIGHEST_PROTOCOL)
                
                mu, std = self.model.predict(genome)
                genome.fitness = self.acquisition(mu, std)
        
        if testing:
            for genome in evaluated_genomes:
                genome.real_fitness = None
        
    def train(self, generation, config): # NOTE: Generation only required for testing
        self.training_set = self.old_training_set + self.training_set
        self.old_training_set = []
        
        df = lambda g1, g2: g1.distance(g2, config.genome_config)
        kernel = RBF(length=1, sigma=1, noise=1, df=df)
        self.model = GaussianProcessModel(kernel, 1, 1, optimize_noise=True)
        
        # NOTE: For testing only
        # with open("data/"+str(generation)+"_train_genomes.pkl", "wb") as output:
        #     pickle.dump(self.training_set, output, pickle.HIGHEST_PROTOCOL)
        self.model.compute(self.training_set, map(lambda g: g.real_fitness, self.training_set), compute_kernel=False)
        with open("data/"+str(generation)+"_model.pkl", "wb") as output:
            pickle.dump(self.model, output, pickle.HIGHEST_PROTOCOL)
        # print(self.model.kernel[0])
        
        self.model.compute(self.training_set, map(lambda g: g.real_fitness, self.training_set))
        self.model.optimize(quiet=True, bounded=True, fevals=200)