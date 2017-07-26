"""Implements the core evolution algorithm."""
from __future__ import print_function

from neat.reporting import ReporterSet
from neat.math_util import mean
from neat.six_util import iteritems, itervalues

from neat.mypy_util import * # pylint: disable=unused-wildcard-import

if MYPY:
    from typing import Callable # pylint: disable=unused-import
    from neat.config import Config # pylint: disable=unused-import
    from neat.reporting import BaseReporter # pylint: disable=unused-import
    from neat.species import DefaultSpeciesSet # pylint: disable=unused-import
    from neat.stagnation import DefaultStagnation # pylint: disable=unused-import
    from neat.reproduction import DefaultReproduction # pylint: disable=unused-import

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

    def __init__(self,
                 config, # type: Config
                 initial_state=None # type: Optional[Tuple[Dict[GenomeKey, KnownGenome], DefaultSpeciesSet, int]]
                 ):
        # type: (...) -> None
        self.reporters = ReporterSet()
        self.config = config
        stagnation = config.stagnation_type(config.stagnation_config, self.reporters) # type: DefaultStagnation # XXX
        self.reproduction = config.reproduction_type(config.reproduction_config,
                                                     self.reporters,
                                                     stagnation) # type: DefaultReproduction # XXX
        if config.fitness_criterion == 'mean': # type: ignore # type: str
            self.fitness_criterion = mean
        elif config.fitness_criterion == 'max': # type: ignore # type: str
            self.fitness_criterion = max # type: ignore
        elif config.fitness_criterion == 'min': # type: ignore # type: str
            self.fitness_criterion = min # type: ignore
        elif not config.no_fitness_termination: # type: ignore # type: bool
            raise RuntimeError(
                "Unexpected fitness_criterion: {0!r}".format(config.fitness_criterion)) # type: ignore

        self.population = {} # type: Dict[GenomeKey, KnownGenome]

        if initial_state is None:
            # Create a population from scratch, then partition into species.
            self.population = self.reproduction.create_new(config.genome_type,
                                                           config.genome_config,
                                                           config.pop_size # type: ignore
                                                           )
            self.species = config.species_set_type(config.species_set_config, self.reporters) # type: DefaultSpeciesSet
            self.generation = 0 # type: int
            self.species.speciate(config, self.population, self.generation)
        else:
            self.population, self.species, self.generation = initial_state

        self.best_genome = None # type: Optional[KnownGenome]

    def add_reporter(self, reporter): # type: (BaseReporter) -> None
        self.reporters.add(reporter)

    def remove_reporter(self, reporter): # type: (BaseReporter) -> None
        self.reporters.remove(reporter)

    def run(self,
            fitness_function, # type: Callable[[List[Tuple[GenomeKey, KnownGenome]], Config], None]
            n=None # type: Optional[int]
            ):
        # type: (...) -> KnownGenome # XXX
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

        if self.config.no_fitness_termination and (n is None): # type: ignore
            raise RuntimeError("Cannot have no generational limit with no fitness termination")

        k = 0 # type: int # c_type: c_uint
        while n is None or k < n:
            k += 1

            self.reporters.start_generation(self.generation)

            # Evaluate all genomes using the user-provided function.
            fitness_function(list(iteritems(self.population)), self.config)

            # Gather and report statistics.
            best = None # type: Optional[KnownGenome]
            for g in itervalues(self.population):
                if best is None or g.fitness > best.fitness:
                    best = g
            self.reporters.post_evaluate(self.config, self.population, self.species, best)

            # Track the best genome ever seen.
            if self.best_genome is None or best.fitness > self.best_genome.fitness: # type: ignore
                self.best_genome = best

            if not self.config.no_fitness_termination: # type: ignore # type: bool
                # End if the fitness threshold is reached.
                fv = self.fitness_criterion(g.fitness for g in itervalues(self.population)) # type: float
                if fv >= self.config.fitness_threshold: # type: ignore
                    self.reporters.found_solution(self.config, self.generation, best)
                    break

            # Create the next generation from the current generation.
            self.population = self.reproduction.reproduce(self.config, self.species,
                                                          self.config.pop_size, # type: ignore # type: int
                                                          self.generation)

            # Check for complete extinction.
            if not self.species.species:
                self.reporters.complete_extinction()

                # If requested by the user, create a completely new population,
                # otherwise raise an exception.
                if self.config.reset_on_extinction: # type: ignore
                    self.population = self.reproduction.create_new(self.config.genome_type,
                                                                   self.config.genome_config,
                                                                   self.config.pop_size # type: ignore
                                                                   )
                else:
                    raise CompleteExtinctionException()

            # Divide the new population into species.
            self.species.speciate(self.config, self.population, self.generation)

            self.reporters.end_generation(self.config, self.population, self.species)

            self.generation += 1

        if self.config.no_fitness_termination: # type: ignore # type: bool
            self.reporters.found_solution(self.config, self.generation, self.best_genome)

        return self.best_genome
