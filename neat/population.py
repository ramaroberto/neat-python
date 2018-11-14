"""Implements the core evolution algorithm."""
from __future__ import print_function

from neat.reporting import ReporterSet
from neat.math_util import mean
from neat.six_util import iteritems, itervalues


class CompleteExtinctionException(Exception):
    pass


class Population(object):
    """
    This class implements the core evolution algorithm:
        1. Evaluate fitness of all genomes.
        2. Check to see if the termination criterion is satisfied; exit if it is.
        3. Generate the next generation from the current population.
        4. Partition the new generation into species based on genetic similarity.
        5. Go to 1.
    """

    def __init__(self, config, initial_state=None):
        self.reporters = ReporterSet()
        self.config = config
        stagnation = config.stagnation_type(config.stagnation_config, self.reporters)
        self.reproduction = config.reproduction_type(config.reproduction_config,
                                                     self.reporters,
                                                     stagnation)
        
        self.surrogate = None
        if config.surrogate_type:
            self.surrogate = config.surrogate_type(config, self.reporters)
            if not self.surrogate.surrogate_config.enabled:
                self.surrogate = None
        
        if config.fitness_criterion == 'max':
            self.fitness_criterion = max
        elif config.fitness_criterion == 'min':
            self.fitness_criterion = min
        elif config.fitness_criterion == 'mean':
            self.fitness_criterion = mean
        elif not config.no_fitness_termination:
            raise RuntimeError(
                "Unexpected fitness_criterion: {0!r}".format(config.fitness_criterion))
                
        if initial_state is None:
            # Create a population from scratch, then partition into species.
            if self.surrogate:
                self.population = self.reproduction.create_new(config.genome_type,
                                                               config.genome_config,
                                                               self.surrogate.surrogate_config.number_initial_samples)
            else:
                self.population = self.reproduction.create_new(config.genome_type,
                                                               config.genome_config,
                                                               config.pop_size)
            self.species = config.species_set_type(config.species_set_config, self.reporters)
            self.generation = 0
            self.species.speciate(config, self.population, self.generation)
        else:
            self.population, self.species, self.generation = initial_state

        self.best_genome = None
        self.evaluations = 0

    def add_reporter(self, reporter):
        self.reporters.add(reporter)

    def remove_reporter(self, reporter):
        self.reporters.remove(reporter)

    def run(self, fitness_function, n=None):
        """
        Runs NEAT's genetic algorithm for at most n generations.  If n
        is None, run until solution is found or extinction occurs.

        The user-provided fitness_function must take only two arguments:
            1. The population as a list of (genome id, genome) tuples.
            2. The current configuration object.

        The return value of the fitness function is ignored, but it must assign
        a Python float to the `fitness` member of each genome.

        The fitness function is free to maintain external state, perform
        evaluations in parallel, etc.

        It is assumed that fitness_function does not modify the list of genomes,
        the genomes themselves (apart from updating the fitness member),
        or the configuration object.
        """
        
        self.resolve_count = 0
        
        if self.config.no_fitness_termination and (n is None):
            raise RuntimeError("Cannot have no generational limit with no fitness termination")

        k = 0
        while n is None or k < n:
            k += 1

            self.reporters.start_generation(self.generation)

            if self.surrogate and self.surrogate.model:
                # If surrogate assistance is enabled, use it instead of evaluation.
                self.surrogate.evaluate(self.population, k, fitness_function, self.config)
            else:
                # Evaluate all genomes using the user-provided function.
                self.evaluations += len(self.population)
                fitness_function(list(iteritems(self.population)), self.config)
                if self.surrogate:
                    self.surrogate.add_to_training(self.population.values())
                    if k == 1:
                        self.surrogate.train(k, self.config)
            print("Total Evaluations: " + str(self.evaluations))
            if self.surrogate:
                print("GP Size: " + str(self.surrogate.samples_length()))
                    
            # Gather and report statistics.
            best = None
            for g in itervalues(self.population):
                if best is None or g.fitness > best.fitness:
                    best = g
            if self.best_genome is None:
                self.best_genome = best
            if not self.surrogate or not self.surrogate.model:
                if best.get_fitness() > self.best_genome.get_fitness():
                    self.best_genome = best
            
            # If surrogate is enabled and conditions matched, train the model.
            if self.surrogate:
                if self.surrogate.model:
                    # If gens_per_infill has been reached, update model.
                    gpi = self.surrogate.surrogate_config.gens_per_infill
                    if k % gpi == 0:
                        # Select the best genomes and update the training set.
                        best_genomes = self.surrogate.update_training(self.species.species, fitness_function, self.config)
                        
                        # Track the best genome ever seen.
                        found_better = False
                        for genome in best_genomes:
                            if self.best_genome is None or \
                                (genome.real_fitness is not None and genome.real_fitness > self.best_genome.real_fitness):
                                self.best_genome = genome
                                self.resolve_count = 0
                                found_better = True
                        if not found_better:
                            self.resolve_count += len(best_genomes)
                        
                        # Add the same number training genomes as population to
                        # stabilize the new model.
                        training_genomes = self.surrogate.get_from_training(len(self.population))
                        self.species.add(self.config, self.generation, training_genomes)
                        
                        # Update the surrogate evaluations with the new model.
                        self.surrogate.evaluate(self.population, k, fitness_function, self.config)
                        
                        # Compute number of evaluations used.
                        self.evaluations += len(best_genomes)
                else:
                    if self.resolve_count == 0 or \
                        self.surrogate.is_training_set_new():
                        self.resolve_count = 0 # is_training_set_new
                        self.surrogate.train(k, self.config)
                print("Resolve Count: ", self.resolve_count, "("+str(self.best_genome.real_fitness)+")")
            
                # If the model is stalled, reset it to produce a resolve.
                if self.surrogate.model \
                    and self.resolve_count >= self.surrogate.surrogate_config.resolve_threshold:
                    self.surrogate.reset()
            
            self.reporters.post_evaluate(self.config, self.population, self.species, best, self.evaluations)
                
            if not self.config.no_fitness_termination:
                # End if the fitness threshold is reached.
                real_fitnesses = []
                for genome in self.population.values():
                    if genome.real_fitness is not None:
                        real_fitnesses.append(genome.real_fitness)
                        
                if real_fitnesses and \
                    self.fitness_criterion(real_fitnesses) >= self.config.fitness_threshold:
                    self.reporters.found_solution(self.config, self.generation, self.best_genome)
                    break
            
            # Create the next generation from the current generation.
            self.population = self.reproduction.reproduce(self.config, self.species,
                                                          self.config.pop_size, self.generation)

            # Check for complete extinction.
            if not self.species.species:
                self.reporters.complete_extinction()

                # If requested by the user, create a completely new population,
                # otherwise raise an exception.
                if self.config.reset_on_extinction:
                    # TODO: Should add SA-NEAT initialization here too, modularize.
                    self.population = self.reproduction.create_new(self.config.genome_type,
                                                                   self.config.genome_config,
                                                                   self.config.pop_size)
                else:
                    raise CompleteExtinctionException()

            # Divide the new population into species.
            self.species.speciate(self.config, self.population, self.generation)

            self.reporters.end_generation(self.config, self.population, self.species)

            self.generation += 1

        if self.config.no_fitness_termination:
            self.reporters.found_solution(self.config, self.generation, self.best_genome)

        return self.best_genome
