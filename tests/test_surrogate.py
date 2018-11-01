import sys
import unittest

# Import the local version of neat instead of the one in the library
import imp, sys
from os import path
local_neat_path = path.dirname(path.dirname(path.abspath(__file__)))
f, pathname, desc = imp.find_module("neat", [local_neat_path])
local_neat = imp.load_module("neat", f, pathname, desc)

from neat.surrogate import DefaultSurrogateModel as DefaultSurrogateModel
import neat

local_dir = path.dirname(__file__)
config_path = path.join(local_dir, 'test_configuration_surrogate')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     neat.surrogate.DefaultSurrogateModel, config_path)

stagnation = config.stagnation_type(config.stagnation_config, None)
reproduction = config.reproduction_type(config.reproduction_config, None, 
    stagnation)

genomes = reproduction.create_new(config.genome_type, config.genome_config, 
    100).values()

def get_genomes_id(genomes):
    return map(lambda g: g.key, genomes)

class TestAddToTrainingComputation(unittest.TestCase):
    def test_add_1(self):
        surrogate = DefaultSurrogateModel(config.surrogate_config, None)
        surrogate.add_to_training([genomes[0]])
        self.assertEqual(get_genomes_id(surrogate.training_set), 
            get_genomes_id([genomes[0]]))
    
    def test_add_multiple(self):
        surrogate = DefaultSurrogateModel(config.surrogate_config, None)
        surrogate.add_to_training(genomes[0:3])
        self.assertEqual(get_genomes_id(surrogate.training_set), 
            get_genomes_id(genomes[0:3]))
    
    def test_add_over_the_limit(self):
        surrogate = DefaultSurrogateModel(config.surrogate_config, None)
        surrogate.max_training_set_size = 20
        surrogate.add_to_training(genomes[0:30])
        self.assertEqual(get_genomes_id(surrogate.training_set), 
            get_genomes_id(genomes[10:30]))
    
    def test_add_when_reseted_1(self):
        surrogate = DefaultSurrogateModel(config.surrogate_config, None)
        surrogate.add_to_training(genomes[0:3])
        
        surrogate.reset()
        surrogate.add_to_training([genomes[3]])
        self.assertEqual(get_genomes_id(surrogate.old_training_set), 
            get_genomes_id(genomes[0:3]))
        self.assertEqual(get_genomes_id(surrogate.training_set), 
            get_genomes_id([genomes[3]]))
    
    def test_add_when_reseted_mult(self):
        surrogate = DefaultSurrogateModel(config.surrogate_config, None)
        surrogate.add_to_training(genomes[0:3])
        
        surrogate.reset()
        surrogate.add_to_training(genomes[3:6])
        self.assertEqual(get_genomes_id(surrogate.old_training_set), 
            get_genomes_id(genomes[0:3]))
        self.assertEqual(get_genomes_id(surrogate.training_set), 
            get_genomes_id(genomes[3:6]))
    
    def test_add_when_reseted_over_the_limit(self):
        surrogate = DefaultSurrogateModel(config.surrogate_config, None)
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
        surrogate = DefaultSurrogateModel(config.surrogate_config, None)
        surrogate.max_training_set_size = 20
        
        surrogate.add_to_training(genomes[0:10])    # Adds 10 genomes
        surrogate.reset()
        surrogate.add_to_training(genomes[10:25])   # Adds 15 genomes
        self.assertEqual(surrogate.is_training_set_new(), False)
        
        surrogate.add_to_training(genomes[25:29])   # Adds 4 genomes
        self.assertEqual(surrogate.is_training_set_new(), False)
        
        surrogate.add_to_training(genomes[29:30])   # Adds 1 genome
        self.assertEqual(surrogate.is_training_set_new(), True)

if __name__ == '__main__':
    unittest.main()