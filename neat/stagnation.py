"""Keeps track of whether species are making progress and helps remove ones that are not."""
import sys

from neat.config import ConfigParameter, DefaultClassConfig
from neat.six_util import iteritems
from neat.math_util import stat_functions

# TODO: Add a method for the user to change the "is stagnant" computation.

class DefaultStagnation(DefaultClassConfig):
    """Keeps track of whether species are making progress and helps remove ones that are not."""
    @classmethod
    def parse_config(cls, param_dict):
        return DefaultClassConfig(param_dict,
                                  [ConfigParameter('species_fitness_func', str, 'mean'),
                                   ConfigParameter('max_stagnation', int, 15),
                                   ConfigParameter('species_elitism', int, 0)])

    def __init__(self, config, reporters):
        # pylint: disable=super-init-not-called
        self.stagnation_config = config

        self.species_fitness_func = stat_functions.get(config.species_fitness_func)
        if self.species_fitness_func is None:
            raise RuntimeError(
                "Unexpected species fitness func: {0!r}".format(config.species_fitness_func))

        self.reporters = reporters

    def update(self, species_set, generation):
        """
        Required interface method. Updates species fitness history information,
        checking for ones that have not improved in max_stagnation generations,
        and - unless it would result in the number of species dropping below the configured
        species_elitism parameter if they were removed, in which case the highest-fitness
        species are spared - returns a list with stagnant species marked for removal.
        """
        species_data = [] # TODO: Move s.XXX, other than fitness, used by stagnation to a
        # s.stagnation_namespace object, added when the species is created; existing
        # uses should be caught by __[get|set]?__ (usual? property?), a DeprecationWarning given,
        # and diverted to the namespace. Ditto for reproduction.
        for sid, s in iteritems(species_set.species):
            if s.stagnation_namespace.fitness_history:
                prev_fitness = max(s.stagnation_namespace.fitness_history)
            else:
                prev_fitness = -sys.float_info.max

            s.stagnation_namespace.fitness = self.species_fitness_func(s.get_fitnesses())
            s.stagnation_namespace.fitness_history.append(s.stagnation_namespace.fitness)
            #s.adjusted_fitness = None # ???
            if prev_fitness is None or s.stagnation_namespace.fitness > prev_fitness:
                s.stagnation_namespace.last_improved = generation

            species_data.append((sid, s))

        # Sort in ascending fitness order.
        species_data.sort(key=lambda x: x[1].stagnation_namespace.fitness)

        result = []
        species_fitnesses = []
        num_non_stagnant = len(species_data)
        for idx, (sid, s) in enumerate(species_data):
            # Override stagnant state if marking this species as stagnant would
            # result in the total number of species dropping below the limit.
            # Because species are in ascending fitness order, less fit species
            # will be marked as stagnant first.
            stagnant_time = generation - s.stagnation_namespace.last_improved
            is_stagnant = False
            if num_non_stagnant > self.stagnation_config.species_elitism:
                is_stagnant = bool(stagnant_time >= self.stagnation_config.max_stagnation)

            if (len(species_data) - idx) <= self.stagnation_config.species_elitism:
                is_stagnant = False

            if is_stagnant:
                num_non_stagnant -= 1

            result.append((sid, s, is_stagnant))
            species_fitnesses.append(s.stagnation_namespace.fitness)

        return result
