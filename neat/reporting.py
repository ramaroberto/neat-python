"""
Makes possible reporter classes,
which are triggered on particular events and may provide information to the user,
may do something else such as checkpointing, or may do both.
"""
from __future__ import division, print_function

import time

from neat.math_util import mean, stdev
from neat.six_util import itervalues, iterkeys

from neat.mypy_util import * # pylint: disable=unused-wildcard-import

if MYPY:
    from neat.species import Species, DefaultSpeciesSet # pylint: disable=unused-import
    from neat.config import Config # pylint: disable=unused-import

# TODO: Add a curses-based reporter.


class ReporterSet(object):
    """
    Keeps track of the set of reporters
    and gives methods to dispatch them at appropriate points.
    """
    def __init__(self):
        self.reporters = [] # type: List[BaseReporter]

    def add(self, reporter): # type: (BaseReporter) -> None
        self.reporters.append(reporter)

    def remove(self, reporter): # type: (BaseReporter) -> None
        self.reporters.remove(reporter)

    def start_generation(self, gen): # type: (int) -> None
        for r in self.reporters:
            r.start_generation(gen)

    def end_generation(self,
                       config, # type: Config
                       population, # type: Dict[GenomeKey, KnownGenome] # XXX
                       species_set # type: DefaultSpeciesSet # XXX
                       ):
        # type: (...) -> None
        for r in self.reporters:
            r.end_generation(config, population, species_set)

    def post_evaluate(self,
                      config, # type: Config
                      population, # type: Dict[GenomeKey, KnownGenome] # XXX
                      species, # type: DefaultSpeciesSet # XXX
                      best_genome # type: KnownGenome # XXX
                      ):
        # type: (...) -> None

        for r in self.reporters:
            r.post_evaluate(config, population, species, best_genome)

    def post_reproduction(self,
                          config, # type: KnownConfig # XXX
                          population, # type: Dict[GenomeKey, KnownGenome] # XXX
                          species_set # type: DefaultSpeciesSet # XXX
                          ):
        # type: (...) -> None
        for r in self.reporters:
            r.post_reproduction(config, population, species_set)

    def complete_extinction(self): # type: () -> None
        for r in self.reporters:
            r.complete_extinction()

    def found_solution(self,
                       config, # type: Config
                       generation, # type: int
                       best # type: KnownGenome # XXX
                       ):
        # type: (...) -> None
        for r in self.reporters:
            r.found_solution(config, generation, best)

    def species_stagnant(self,
                         sid, # type: SpeciesKey
                         species # type: Species
                         ):
        # type: (...) -> None
        for r in self.reporters:
            r.species_stagnant(sid, species)

    def info(self, msg): # type: (str) -> None
        for r in self.reporters:
            r.info(msg)


class BaseReporter(object):
    """Definition of the reporter interface expected by ReporterSet."""
    def start_generation(self, generation): # type: (int) -> None
        pass

    def end_generation(self,
                       config, # type: Config
                       population, # type: Dict[GenomeKey, KnownGenome] # XXX
                       species_set # type: DefaultSpeciesSet # XXX
                       ):
        # type: (...) -> None
        pass

    def post_evaluate(self,
                      config, # type: Config
                      population, # type: Dict[GenomeKey, KnownGenome] # XXX
                      species, # type: DefaultSpeciesSet # XXX
                      best_genome # type: KnownGenome # XXX
                      ):
        # type: (...) -> None
        pass

    def post_reproduction(self,
                          config, # type: KnownConfig # XXX
                          population, # type: Dict[GenomeKey, KnownGenome] # XXX
                          species_set # type: DefaultSpeciesSet # XXX
                          ):
        # type: (...) -> None
        pass

    def complete_extinction(self): # type: () -> None
        pass

    def found_solution(self,
                       config, # type: Config
                       generation, # type: int
                       best # type: KnownGenome # XXX
                       ):
        # type: (...) -> None
        pass

    def species_stagnant(self,
                         sid, # type: SpeciesKey
                         species # type: Species
                         ):
        # type: (...) -> None
        pass

    def info(self, msg): # type: (str) -> None
        pass


class StdOutReporter(BaseReporter):
    def __init__(self, show_species_detail): # type: (bool) -> None
        self.show_species_detail = show_species_detail
        self.generation = None # type: Optional[int]
        self.generation_start_time = None # type: Optional[float]
        self.generation_times = [] # type: List[float]
        self.num_extinctions = 0 # type: int # c_type: c_uint

    def start_generation(self, generation): # type: (int) -> None
        self.generation = generation
        print('\n ****** Running generation {0} ****** \n'.format(generation))
        self.generation_start_time = time.time()

    def end_generation(self,
                       config, # type: Config
                       population, # type: Dict[GenomeKey, KnownGenome] # XXX
                       species_set # type: DefaultSpeciesSet # XXX
                       ):
        # type: (...) -> None
        ng = len(population) # type: int # c_type: c_uint
        ns = len(species_set.species) # type: int # c_type: c_uint
        if self.show_species_detail:
            print('Population of {0:d} members in {1:d} species:'.format(ng, ns))
            sids = list(iterkeys(species_set.species)) # type: List[SpeciesKey]
            sids.sort()
            print("   ID   age  size  fitness  adj fit  stag")
            print("  ====  ===  ====  =======  =======  ====")
            for sid in sids:
                s = species_set.species[sid]
                a = self.generation - s.created
                n = len(s.members)
                f = "--" if s.fitness is None else "{:.1f}".format(s.fitness)
                af = "--" if s.adjusted_fitness is None else "{:.3f}".format(s.adjusted_fitness)
                st = self.generation - s.last_improved
                print(
                    "  {: >4}  {: >3}  {: >4}  {: >7}  {: >7}  {: >4}".format(sid, a, n, f, af, st))
        else:
            print('Population of {0:d} members in {1:d} species'.format(ng, ns))

        elapsed = time.time() - self.generation_start_time # type: float
        self.generation_times.append(elapsed)
        self.generation_times = self.generation_times[-10:]
        average = sum(self.generation_times) / len(self.generation_times) # type: float
        print('Total extinctions: {0:d}'.format(self.num_extinctions))
        if len(self.generation_times) > 1:
            print("Generation time: {0:.3f} sec ({1:.3f} average)".format(elapsed, average))
        else:
            print("Generation time: {0:.3f} sec".format(elapsed))

    def post_evaluate(self,
                      config, # type: Config
                      population, # type: Dict[GenomeKey, KnownGenome] # XXX
                      species, # type: DefaultSpeciesSet # XXX
                      best_genome # type: KnownGenome # XXX
                      ):
        # type: (...) -> None
        # pylint: disable=no-self-use
        fitnesses = [c.fitness for c in itervalues(population)] # type: List[float]
        fit_mean = mean(fitnesses) # type: float
        fit_std = stdev(fitnesses) # type: float
        best_species_id = species.get_species_id(best_genome.key) # type: SpeciesKey
        print('Population\'s average fitness: {0:3.5f} stdev: {1:3.5f}'.format(fit_mean, fit_std))
        print(
            'Best fitness: {0:3.5f} - size: {1!r} - species {2} - id {3}'.format(best_genome.fitness,
                                                                                 best_genome.size(),
                                                                                 best_species_id,
                                                                                 best_genome.key))

    def complete_extinction(self): # type: () -> None
        self.num_extinctions += 1
        print('All species extinct.')

    def found_solution(self,
                       config, # type: Config
                       generation, # type: int
                       best # type: KnownGenome # XXX
                       ):
        # type: (...) -> None
        print('\nBest individual in generation {0} meets fitness threshold - complexity: {1!r}'.format(
            self.generation, best.size()))

    def species_stagnant(self, sid, species): # type: (SpeciesKey, Species) -> None
        if self.show_species_detail:
            print("\nSpecies {0} with {1} members is stagnated: removing it".format(sid, len(species.members)))

    def info(self, msg): # type: (str) -> None
        print(msg)
