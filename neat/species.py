"""Divides the population into species based on genomic distances."""
from __future__ import division

import math
import sys
import warnings

from argparse import Namespace
from itertools import count

from neat.math_util import mean, stdev, tmean
from neat.six_util import iteritems, iterkeys, itervalues
from neat.config import ConfigParameter, DefaultClassConfig

class Species(object):
    def __init__(self, key, generation):
        self.key = key
        self.created = generation
        self.representative = None
        self.members = {}
        self.reproduction_namespace = Namespace()
        self.reproduction_namespace.adjusted_fitness = None
        self.stagnation_namespace = Namespace()
        self.stagnation_namespace.fitness = None
        self.stagnation_namespace.last_improved = generation
        self.stagnation_namespace.fitness_history = []

    def getAdjustedFitness(self): # pragma: no cover
        """
        Backwards compatibility wrapper for species.adjusted_fitness;
        use species.reproduction_namespace.adjusted_fitness instead.
        """
        warnings.warn("Use species.reproduction_namespace for adjusted_fitness",
                      DeprecationWarning, stacklevel=2)
        return self.reproduction_namespace.adjusted_fitness

    def setAdjustedFitness(self,value): # pragma: no cover
        if not isinstance(value, float):
            raise TypeError(
                "Adjusted_fitness ({0!r}) should be a float, not {1!s}".format(
                    value, type(value)))
        warnings.warn("Use species.reproduction_namespace for adjusted_fitness",
                      DeprecationWarning, stacklevel=2)
        self.reproduction_namespace.adjusted_fitness = value

    def delAdjustedFitness(self): # pragma: no cover
        warnings.warn("Use species.reproduction_namespace for adjusted_fitness",
                      DeprecationWarning, stacklevel=2)
        self.reproduction_namespace.adjusted_fitness = None
    
    adjusted_fitness = property(getAdjustedFitness,
                                setAdjustedFitness,
                                delAdjustedFitness)

    def getFitness(self): # pragma: no cover
        """
        Backwards compatibility wrapper for species.fitness;
        use species.stagnation_namespace.fitness instead.
        """
        warnings.warn("Use species.stagnation_namespace for fitness",
                      DeprecationWarning, stacklevel=2)
        return self.stagnation_namespace.fitness

    def setFitness(self,value): # pragma: no cover
        if not isinstance(value, float):
            raise TypeError(
                "Fitness ({0!r}) should be a float, not {1!s}".format(
                    value, type(value)))
        warnings.warn("Use species.stagnation_namespace for fitness",
                      DeprecationWarning, stacklevel=2)
        self.stagnation_namespace.fitness = value

    def delFitness(self): # pragma: no cover
        warnings.warn("Use species.stagnation_namespace for fitness",
                      DeprecationWarning, stacklevel=2)
        self.stagnation_namespace.fitness = None

    fitness = property(getFitness,
                       setFitness,
                       delFitness)

    def getLastImproved(self): # pragma: no cover
        """
        Backwards compatibility wrapper for species.last_improved;
        use species.stagnation_namespace.last_improved instead.
        """
        warnings.warn("Use species.stagnation_namespace for last_improved",
                      DeprecationWarning, stacklevel=2)
        return self.stagnation_namespace.last_improved

    def setLastImproved(self, value): # pragma: no cover
        if not isinstance(value, int):
            raise TypeError(
                "Last_improved ({0!r}) should be an int, not {1!s}".format(
                    value, type(value)))
        warnings.warn("Use species.stagnation_namespace for last_improved",
                      DeprecationWarning, stacklevel=2)
        self.stagnation_namespace.last_improved = value
    last_improved = property(getLastImproved,
                             setLastImproved)

    @property
    def fitness_history(self): # pragma: no cover
        """Sole method available for this one, given is a list..."""
        warnings.warn("Use species.stagnation_namespace for fitness_history",
                      stacklevel=2)
        return self.stagnation_namespace.fitness_history

    def update(self, representative, members):
        self.representative = representative
        self.members = members

    def get_fitnesses(self):
        return [m.fitness for m in itervalues(self.members)]


class GenomeDistanceCache(object):
    def __init__(self, config):
        self.distances = {}
        self.config = config
        self.hits = 0
        self.misses = 0

    def __call__(self, genome0, genome1):
        g0 = genome0.key
        g1 = genome1.key
        d = self.distances.get((g0, g1))
        if d is None:
            # Distance is not already computed.
            d = genome0.distance(genome1, self.config)
            self.distances[g0, g1] = d
            self.distances[g1, g0] = d
            self.misses += 1
        else:
            self.hits += 1

        return d

class DefaultSpeciesSet(DefaultClassConfig):
    """ Encapsulates the default speciation scheme. """

    def __init__(self, config, reporters, reproduction=None):
        # pylint: disable=super-init-not-called
        self.species_set_config = config
        self.reporters = reporters
        self.reproduction = reproduction
        self.indexer = count(1)
        self.species = {}
        self.genome_to_species = {}
        self.orig_compatibility_threshold = config.compatibility_threshold
        if ((config.compatibility_threshold_adjust.lower() != 'fixed') or
            (reproduction is not None)):
            self.min_pop_seen = 10**sys.float_info.dig
            self.max_pop_seen = 0
            if reproduction is None: # pragma: no cover
                raise RuntimeError(
                    "Need reproduction instance for species info (threshold_adjust {0!s})".format(
                        self.compatibility_threshold_adjust))
            self.threshold_adjust_dict = reproduction.get_species_size_info()
            if ((config.compatibility_threshold_adjust.lower() != 'fixed') or
                (self.threshold_adjust_dict['genome_config'] is not None)):
                if self.threshold_adjust_dict['genome_config'] is None: # pragma: no cover
                    raise RuntimeError(
                        "Need genome_config instance for species info (threshold_adjust {0!s})".format(
                            self.compatibility_threshold_adjust))
                self.threshold_adjust_dict.update(
                    self.threshold_adjust_dict['genome_config'].get_compatibility_info())
                # below is based on configuration values from Stanley's website as
                # compared to advised adjustment size.
                self.base_threshold_adjust = 0.3*max(
                    self.threshold_adjust_dict['weight_coefficient'],
                    (self.threshold_adjust_dict['disjoint_coefficient']/2),
                    (self.orig_compatibility_threshold/6))
            # for max_stagnation
            self.threshold_adjust_dict.update(
                self.threshold_adjust_dict['stagnation'].get_stagnation_info())

    @classmethod
    def parse_config(cls, param_dict):
        return DefaultClassConfig(param_dict,
                                  [ConfigParameter('compatibility_threshold', float),
                                   ConfigParameter('compatibility_threshold_adjust',
                                                   str, 'fixed'),
                                   ConfigParameter('desired_species_num',
                                                   float, 0.0, default_ok=True)])

    def find_desired_num_species(self, pop_size_high, pop_size_low): # DOCUMENT!
        if self.species_set_config.desired_species_num >= 2: # NEED TEST!
            max_num_usable = pop_size_high/self.threshold_adjust_dict['min_size']
            if max_num_usable < 2:
                raise ValueError(
                    "Pop_size {0:n} is too low for effective min species size {1:n}".format(
                        pop_size_high, self.threshold_adjust_dict['min_size']))
##            max_num_usable = max(max_num_usable,2)
            if self.species_set_config.desired_species_num > max_num_usable: # NEED TEST!
                warnings.warn(
                    "Desired_species_num {0:n} is too high for pop_size {1:n};".format(
                        self.species_set_config.desired_species_num,pop_size_high)
                    + " adjusting to max {0:n}".format(max_num_usable))
                self.species_set_config.desired_species_num = max_num_usable
                sys.stderr.flush()
            to_return = self.species_set_config.desired_species_num
            # the below with floor and ceil results in:
            # A. If input is an integer, _high is +1 and _low is -1
            # B. If input is not an integer, _high is ceil and _low is floor
            to_return_high = min(math.ceil(max_num_usable),
                                 math.floor(max(math.sqrt(pop_size_high),to_return)+1.0))
            to_return_low = max(2, math.ceil(min(math.sqrt(pop_size_low),to_return)-1.0))
            return (to_return_high,to_return_low)
        elif self.species_set_config.desired_species_num != 0:
            warnings.warn(
                "Desired_species_num of {0:n} not valid; treating as 0 (autoconfigure)".format(
                    self.species_set_config.desired_species_num))
            self.species_set_config.desired_species_num = 0

        poss_num_high = math.floor((pop_size_high/
                                    self.threshold_adjust_dict['min_OK_size'])+1.0)
        if poss_num_high < 2: # NEED TEST!
            raise ValueError(
                "Pop_size {0:n} is too low to determine desired num species;".format(pop_size_high)
                + " need minimum of {0:n} given min_good_size {1:n}".format(
                    (2*self.threshold_adjust_dict['min_OK_size']),
                    self.threshold_adjust_dict['min_OK_size']))
        poss_num_low = max(2,math.ceil((pop_size_low/
                                        self.threshold_adjust_dict['min_good_size'])-1.0))
        return (poss_num_high,poss_num_low)

    def adjust_compatibility_threshold(self, increase, curr_tmean, max_rep_dist): # DOCUMENT!
        old_threshold = self.species_set_config.compatibility_threshold
        if increase:
            mult_threshold = 1.05*old_threshold
            add_threshold = old_threshold + self.base_threshold_adjust
            if old_threshold >= self.orig_compatibility_threshold:
                new_threshold = min(mult_threshold,add_threshold)
            else:
                new_threshold = max(min(mult_threshold,add_threshold),
                                    min(self.orig_compatibility_threshold,
                                        max(mult_threshold,add_threshold)))
            if (curr_tmean
                < self.orig_compatibility_threshold) and (new_threshold
                                                          < curr_tmean):
                new_threshold = min(curr_tmean,((new_threshold
                                                 + self.orig_compatibility_threshold)/2))
            which = 'increased'
        else:
            div_threshold = old_threshold/1.05
            sub_threshold = old_threshold - self.base_threshold_adjust
            if old_threshold <= self.orig_compatibility_threshold:
                new_threshold = max(div_threshold,sub_threshold)
            else:
                new_threshold = min(max(div_threshold,sub_threshold),
                                    max(self.orig_compatibility_threshold,
                                        min(div_threshold,sub_threshold)))

            new_threshold = max(new_threshold,min(max_rep_dist,old_threshold))
            which = 'decreased'
        self.species_set_config.compatibility_threshold = new_threshold
        self.reporters.info(
            "Compatibility threshold (orig {0:n}) {1!s} by {2:n} to {3:n} (from {4:n})".format(
                self.orig_compatibility_threshold,
                which,
                abs(self.species_set_config.compatibility_threshold-old_threshold),
                self.species_set_config.compatibility_threshold,
                old_threshold))

    def speciate(self, config, population, generation):
        """
        Place genomes into species by genetic similarity.

        Note that this method assumes the current representatives of the species are from the old
        generation, and that after speciation has been performed, the old representatives should be
        dropped and replaced with representatives from the new generation.  If you violate this
        assumption, you should make sure other necessary parts of the code are updated to reflect
        the new behavior.
        """
        if not isinstance(population, dict): # TEST NEEDED!
            raise TypeError("Population ({0!r}) should be a dict, not {1!s}".format(
                population, type(population)))

        # Find the best representatives for each existing species.
        unspeciated = set(iterkeys(population))
        distances = GenomeDistanceCache(config.genome_config)
        gid_min_dist = dict((gid, sys.float_info.max) for gid in unspeciated)
        new_representatives = {}
        new_members = {}
        species_list = list(iterkeys(self.species))
        species_list.sort(reverse=True,
                          key=lambda x: self.species[x].reproduction_namespace.adjusted_fitness)
        max_rep_dist = 0.0
        for sid in species_list:
            s = self.species[sid]
            candidates = []
            for gid in unspeciated:
                g = population[gid]
                d = distances(s.representative, g)
                candidates.append((d, g))
                gid_min_dist[gid] = min(gid_min_dist[gid], d)

            # The new representative is the genome closest to the current representative.
            rdist, new_rep = min(candidates, key=lambda x: x[0])
            max_rep_dist = max(max_rep_dist, rdist)
            new_rid = new_rep.key
            if rdist >= self.species_set_config.compatibility_threshold: # pragma: no cover
                warnings.warn(
                    "Closest genome {0:n} to species {1:n}: dist {2:n} (thresh {3:n})".format(
                        new_rid, sid, rdist, self.species_set_config.compatibility_threshold))
            new_representatives[sid] = new_rid
            new_members[sid] = [new_rid]
            unspeciated.remove(new_rid)

        self.reporters.info(
            "Furthest rep from old species representatives was at {0:n} ({1:n}%)".format(
                max_rep_dist,
                (100.0*max_rep_dist/self.species_set_config.compatibility_threshold)))

        unspeciated_list = list(unspeciated)
        # Putting most distant from any others first, to serve as clustering seeds
        unspeciated_list.sort(reverse=True, key=lambda x: gid_min_dist[x])

        # Partition population into species based on genetic similarity.
        for gid in unspeciated_list:
            g = population[gid]

            # Find the species with the most similar representative.
            candidates = []
            for sid, rid in iteritems(new_representatives):
                rep = population[rid]
                d = distances(rep, g)
                if d < self.species_set_config.compatibility_threshold:
                    candidates.append((d, sid))

            if candidates:
                ignored_sdist, sid = min(candidates, key=lambda x: x[0])
                new_members[sid].append(gid)
            else:
                # No species is similar enough, create a new species, using
                # this genome as its representative.
                sid = next(self.indexer)
                new_representatives[sid] = gid
                new_members[sid] = [gid]

        # Update species collection based on new speciation.
        self.genome_to_species = {}
        for sid, rid in iteritems(new_representatives):
            s = self.species.get(sid)
            if s is None:
                s = Species(sid, generation)
                self.species[sid] = s

            members = new_members[sid]
            for gid in members:
                self.genome_to_species[gid] = sid

            member_dict = dict((gid, population[gid]) for gid in members)
            s.update(population[rid], member_dict)

        gdmean = mean(itervalues(distances.distances))
        gdtmean = tmean(itervalues(distances.distances))
        gdstdev = stdev(itervalues(distances.distances))
        self.reporters.info(
            'Mean genetic distance {0:n}, 50% trimmed mean {1:n}, standard deviation {2:n}'.format(
                gdmean, gdtmean, gdstdev))

        if self.species_set_config.compatibility_threshold_adjust.lower() == 'number':
            if generation < 1:
                self.reporters.info(
                    "Min_size is {0:n}, min_good_size is {1:n}".format(
                        self.threshold_adjust_dict['min_size'],
                        self.threshold_adjust_dict['min_good_size']))
            self.min_pop_seen = min(self.min_pop_seen,len(population))
            self.max_pop_seen = max(self.max_pop_seen,len(population))
            desired_num_species_high, desired_num_species_low = self.find_desired_num_species(
                self.max_pop_seen, self.min_pop_seen)
            if len(self.species) < desired_num_species_low:
                self.reporters.info(
                    "Species num {0:n} below desired minimum {1:n}".format(
                        len(self.species), desired_num_species_low))
                self.adjust_compatibility_threshold(increase=False,
                                                    curr_tmean=gdtmean,
                                                    max_rep_dist=max_rep_dist)
            elif len(self.species) > desired_num_species_high:
                self.reporters.info(
                    "Species num {0:n} above desired maximum {1:n}".format(
                        len(self.species), desired_num_species_high))
                if (('max_stagnation' not in self.threshold_adjust_dict)
                    or (generation >= self.threshold_adjust_dict['max_stagnation'])
                    or (self.species_set_config.compatibility_threshold <
                        self.orig_compatibility_threshold)
                    or (self.species_set_config.compatibility_threshold <=
                        max_rep_dist)):
                    self.adjust_compatibility_threshold(increase=True,
                                                        curr_tmean=gdtmean,
                                                        max_rep_dist=max_rep_dist)
            elif ((self.species_set_config.compatibility_threshold <= max_rep_dist)
                  and (len(self.species) > desired_num_species_low)):
                self.adjust_compatibility_threshold(increase=True,
                                                    curr_tmean=gdtmean,
                                                    max_rep_dist=max_rep_dist)
        elif self.species_set_config.compatibility_threshold_adjust.lower() != 'fixed':
            raise ValueError(
                "Unknown compatibility_threshold_adjust {!r}".format(
                    self.species_set_config.compatibility_threshold_adjust))

    def get_species_id(self, individual_id):
        return self.genome_to_species[individual_id]

    def get_species(self, individual_id): # NEED TEST!
        sid = self.genome_to_species[individual_id]
        return self.species[sid]
