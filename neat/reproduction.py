"""
Handles creation of genomes, either from scratch or by sexual or
asexual reproduction from parents.
"""
from __future__ import division, print_function

import math
import random

from itertools import count
from sys import stderr, float_info

from neat.config import ConfigParameter, DefaultClassConfig
from neat.math_util import mean, NORM_EPSILON
from neat.six_util import iteritems, itervalues

# TODO: Provide some sort of optional cross-species performance criteria, which
# are then used to control stagnation and possibly the mutation rate
# configuration. This scheme should be adaptive so that species do not evolve
# to become "cautious" and only make very slow progress.

class DefaultReproduction(DefaultClassConfig):
    """
    Implements the default NEAT-python reproduction scheme:
    explicit fitness sharing with fixed-time species stagnation.
    """

    @classmethod
    def parse_config(cls, param_dict):
        return DefaultClassConfig(param_dict,
                                  [ConfigParameter('elitism', int, 0),
                                   ConfigParameter('min_for_elitism', int, 0),
                                   ConfigParameter('survival_threshold', float, 0.2),
                                   ConfigParameter('min_species_size', int, 0),
                                   ConfigParameter('fitness_min_divisor', float, 1.0),
                                   ConfigParameter('selection_tournament_size', int, 2),
                                   ConfigParameter('crossover_prob', float, 0.75),
                                   ConfigParameter('minimum_species', int, 0),
                                   ConfigParameter('filter_bad_genomes', bool, False),
                                   ConfigParameter('square_adjusted_fitness', bool, False)])

    def __init__(self, config, reporters, stagnation):
        # pylint: disable=super-init-not-called
        self.reproduction_config = config
        self.reporters = reporters
        self.genome_indexer = count(1)
        self.stagnation = stagnation
        self.ancestors = {}

        if config.fitness_min_divisor < 0.0:
            raise RuntimeError(
                "Fitness_min_divisor cannot be negative ({0:n})".format(
                    config.fitness_min_divisor))
        elif config.fitness_min_divisor == 0.0:
            config.fitness_min_divisor = NORM_EPSILON
        elif config.fitness_min_divisor < float_info.epsilon:
            print("Fitness_min_divisor {0:n} is too low; increasing to {1:n}".format(
                config.fitness_min_divisor,float_info.epsilon), file=stderr)
            stderr.flush()
            config.fitness_min_divisor = float_info.epsilon


    def create_new(self, genome_type, genome_config, num_genomes):
        new_genomes = {}
        for i in range(num_genomes):
            key = next(self.genome_indexer)
            g = genome_type(key)
            g.configure_new(genome_config)
            new_genomes[key] = g
            self.ancestors[key] = tuple()

        return new_genomes

    @staticmethod
    def compute_spawn(fitnesses, previous_sizes, pop_size, min_species_size, square_fitness=False):
        """Compute the proper number of offspring per species (proportional to fitness)."""
        if square_fitness:
            fitnesses = map(lambda v: v**2, fitnesses)
        af_sum = sum(fitnesses)

        spawn_amounts = []
        for af, ps in zip(fitnesses, previous_sizes):
            if af_sum > 0:
                s = max(min_species_size, af / af_sum * pop_size)
            else:
                s = min_species_size

            d = (s - ps) * 0.5
            c = int(round(d))
            spawn = ps
            if abs(c) > 0:
                spawn += c
            elif d > 0:
                spawn += 1
            elif d < 0:
                spawn -= 1

            spawn_amounts.append(spawn)

        # Normalize the spawn amounts so that the next generation is roughly
        # the population size requested by the user.
        total_spawn = sum(spawn_amounts)
        norm = pop_size / total_spawn
        spawn_amounts = [max(min_species_size, int(round(n * norm))) for n in spawn_amounts]

        return spawn_amounts

    def reproduce(self, config, species, pop_size, generation):
        """
        Handles creation of genomes, either from scratch or by sexual or
        asexual reproduction from parents.
        """
        # TODO: I don't like this modification of the species and stagnation objects,
        # because it requires internal knowledge of the objects.

        # Filter out stagnated species, collect the set of non-stagnated
        # species members, and compute their average adjusted fitness.
        # The average adjusted fitness scheme (normalized to the interval
        # [0, 1]) allows the use of negative fitness values without
        # interfering with the shared fitness scheme.
        all_fitnesses = []
        remaining_species = []
        for stag_sid, stag_s, stagnant in self.stagnation.update(species, generation):
            if stagnant:
                self.reporters.species_stagnant(stag_sid, stag_s)
            else:
                all_fitnesses.extend(m.get_fitness() for m in itervalues(stag_s.members))
                remaining_species.append(stag_s)
        # The above comment was not quite what was happening - now getting fitnesses
        # only from members of non-stagnated species.

        # No species left.
        old_species = species.species.values()
        random.shuffle(old_species)
        if not remaining_species:
            if self.reproduction_config.minimum_species <= 0:
                species.species = {}
                return {} # was []
            while len(old_species) > 0 and \
                len(remaining_species) < self.reproduction_config.minimum_species:
                s = old_species.pop(0)
                all_fitnesses.extend(m.get_fitness() for m in itervalues(s.members))
                remaining_species.append(s)
                

        # Find minimum/maximum fitness across the entire population, for use in
        # species adjusted fitness computation.
        min_fitness = min(all_fitnesses)
        max_fitness = max(all_fitnesses)
        # Do not allow the fitness range to be zero, as we divide by it below.
        fitness_range = max(self.reproduction_config.fitness_min_divisor, max_fitness - min_fitness)
        for afs in remaining_species:
            # Compute adjusted fitness.
            msf = mean([m.get_fitness() for m in itervalues(afs.members)])
            af = (msf - min_fitness) / fitness_range
            afs.adjusted_fitness = af

        adjusted_fitnesses = [s.adjusted_fitness for s in remaining_species]
        avg_adjusted_fitness = mean(adjusted_fitnesses) # type: float
        self.reporters.info("Average adjusted fitness: {:.3f}".format(avg_adjusted_fitness))

        # Compute the number of new members for each species in the new generation.
        previous_sizes = [len(s.members) for s in remaining_species]
        min_species_size = self.reproduction_config.min_species_size
        # Isn't the effective min_species_size going to be max(min_species_size,
        # self.reproduction_config.elitism)? That would probably produce more accurate tracking
        # of population sizes and relative fitnesses... doing. TODO: document.
        min_species_size = max(min_species_size,self.reproduction_config.elitism)
        spawn_amounts = self.compute_spawn(adjusted_fitnesses, previous_sizes,
                                           pop_size, min_species_size, square_fitness=self.reproduction_config.square_adjusted_fitness)

        new_population = {}
        species.species = {}
        for spawn, s in zip(spawn_amounts, remaining_species):
            old_members = list(iteritems(s.members))
            s.members = {}
            species.species[s.key] = s

            # Sort members in order of descending fitness.
            old_members.sort(reverse=True, key=lambda m: m[1].get_fitness())

            # Transfer elites to new generation.
            if self.reproduction_config.min_for_elitism <= len(old_members) and \
                self.reproduction_config.elitism > 0:
                for i, m in old_members[:self.reproduction_config.elitism]:
                    if spawn <= 0:
                        break
                    new_population[i] = m
                    spawn -= 1
            if spawn <= 0:
                continue

            # Only use the survival threshold fraction to use as parents for the
            # next generation.
            repro_cutoff = int(math.ceil(self.reproduction_config.survival_threshold *
                                         len(old_members)))
            # Try to use at least two parents no matter what the threshold 
            # fraction result is. If the bad genomes' filter is enabled, 
            # additionaly delete all the genomes matching the minimum fitness.
            repro_cutoff = max(repro_cutoff, min(2, len(old_members)))
            if self.reproduction_config.filter_bad_genomes and \
                old_members[0][1].get_fitness() > min_fitness and repro_cutoff > 1:
                while abs(old_members[repro_cutoff-1][1].get_fitness() - min_fitness) < 1e-10:
                    repro_cutoff -= 1
            
            # Trim the population with the cutoff value.
            old_members = old_members[:repro_cutoff]

            # Randomly choose parents and produce the number of offspring
            # allotted to the species.
            while spawn > 0:
                spawn -= 1
                gid = next(self.genome_indexer)
                child = config.genome_type(gid)
                
                # Perform tournament selection on old members in order to choose
                # parents. Note that if the parents are not distinct, crossover 
                # will produce a genetically identical clone of the parent 
                # (but with a different ID).
                if len(old_members) > 1:
                    ip1 = min(random.sample(xrange(len(old_members)), 
                        min(len(old_members), self.reproduction_config.selection_tournament_size)))
                    ip2 = min(random.sample(xrange(len(old_members)), 
                        min(len(old_members), self.reproduction_config.selection_tournament_size)))
                        
                    r = random.random()
                    if r < self.reproduction_config.crossover_prob:
                        parent1_id, parent1 = old_members[ip1]
                        parent2_id, parent2 = old_members[ip2]
                    else:
                        parent1_id, parent1 = old_members[min(ip1, ip2)]
                        parent2_id, parent2 = (parent1_id, parent1)
                else:
                    parent1_id, parent1 = old_members[0]
                    parent2_id, parent2 = (parent1_id, parent1)
                    
                child.configure_crossover(parent1, parent2, config.genome_config)
                self.ancestors[gid] = (parent1_id, parent2_id)
                
                child.mutate(config.genome_config)
                new_population[gid] = child

        return new_population
