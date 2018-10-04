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

    def reproduce(self, config, species, pop_size, generation):
        """
        Handles creation of genomes, either from scratch or by sexual or
        asexual reproduction from parents.
        """
        
        remaining_species = []
        fitness_total = 0.0
        for sid, s, is_stagnant in self.stagnation.update(species, generation):
            # Calculate the average fitness of the species.
            fitsum = 0.0
            for m in itervalues(s.members):
                if self.reproduction_config.square_adjusted_fitness:
                    # fitsum += (1.0 + fitness_floor + m.fitness) ** 2.0
                    fitsum += (1.0 + m.fitness) ** 2
                else:
                    fitsum += m.fitness
            
            s.adjusted_fitness = 0.0
            if len(s.members) > 0 and fitsum > 1e-10:
                s.adjusted_fitness = float(fitsum) / len(s.members)
            
            # Only keep the non-stagnant doing-something species.
            if not is_stagnant and s.adjusted_fitness > 0.1:
                remaining_species.append(s)
                fitness_total += s.adjusted_fitness

        # No species left, if minimum is set shuffle and grab some of them.
        if not remaining_species and \
            self.reproduction_config.minimum_species > 0:
            old_species = species.species.values()
            random.shuffle(old_species)
            if self.reproduction_config.minimum_species <= 0:
                species.species = {}
                return {}
            while len(old_species) > 0 and \
                len(remaining_species) < self.reproduction_config.minimum_species:
                s = old_species.pop(0)
                remaining_species.append(s)
                fitness_total += s.adjusted_fitness
        
        # Avoid division by zero, in the case the generation is bad.
        if fitness_total < 1e-10:
            fitness_total = 1.0

        # Compute the number of new members for each species in the new generation.
        min_species_size = self.reproduction_config.min_species_size
        pop_to_accommodate = pop_size - len(remaining_species) * min_species_size
        
        # Calculate the adjusted fitness and keep track of the remainders of the
        # integer division.
        spawn_amounts = []
        remainders = []
        for i, s in enumerate(remaining_species):
            s.adjusted_fitness /= fitness_total
            earned_spawn = s.adjusted_fitness * pop_to_accommodate
            spawn = math.floor(earned_spawn)
            spawn_amounts.append(spawn)
            remainders.append((i, earned_spawn - spawn))
        
        # Use the remainders to assign the excess spawns to the most rightful
        # species.
        pop_to_accommodate -= sum(spawn_amounts)
        remainders.sort(key=lambda e: e[1], reverse=True)
        for i, remainder in remainders:
            if pop_to_accommodate <= 0:
                break
            spawn_amounts[i] += 1
            pop_to_accommodate -= 1
            
        
        new_population = {}
        species.species = {}
        for spawn, s in zip(spawn_amounts, remaining_species):
            old_members = list(iteritems(s.members))
            s.members = {}
            species.species[s.key] = s

            # Sort members in order of descending fitness.
            old_members.sort(reverse=True, key=lambda x: x[1].fitness)

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
            # fraction result is.
            repro_cutoff = max(repro_cutoff, min(2, len(old_members)))
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
                    # TODO(robertorama): ip1 and ip2 could happen to be the 
                    # same. This can be avoided by sampling all the members at 
                    # the same time.
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
