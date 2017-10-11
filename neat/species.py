"""Divides the population into species based on genomic distances."""
from __future__ import division

import math
import sys
import warnings

# Namespace is used for species so reproduction,
# stagnation won't trample over each other's attribute names
from argparse import Namespace
from itertools import count

from neat.math_util import mean, stdev, tmean, NORM_EPSILON
from neat.six_util import iteritems, iterkeys, itervalues
from neat.config import ConfigParameter, DefaultClassConfig

class Species(object):
    def __init__(self, key, generation):
        self.key = key
        self.created = generation
        self.representative = None
        self.members = {}
        self.at_bottom = False # for relative fitnesses
        self.reproduction_namespace = Namespace()
        self.reproduction_namespace.adjusted_fitness = None
        self.reproduction_namespace.orig_fitness = None
        self.stagnation_namespace = Namespace()
        self.stagnation_namespace.fitness = None
        self.stagnation_namespace.last_improved = generation
        self.stagnation_namespace.fitness_history = []

    def _getAdjustedFitness(self): # pragma: no cover
        """
        Backwards compatibility wrapper for species.adjusted_fitness;
        use species.reproduction_namespace.adjusted_fitness instead.
        """
        warnings.warn("Use species.reproduction_namespace for adjusted_fitness",
                      DeprecationWarning, stacklevel=2)
        return self.reproduction_namespace.adjusted_fitness

    def _setAdjustedFitness(self,value): # pragma: no cover
        if not isinstance(value, float):
            raise TypeError(
                "Adjusted_fitness ({0!r}) should be a float, not {1!s}".format(
                    value, type(value)))
        warnings.warn("Use species.reproduction_namespace for adjusted_fitness",
                      DeprecationWarning, stacklevel=2)
        self.reproduction_namespace.adjusted_fitness = value

    def _delAdjustedFitness(self): # pragma: no cover
        warnings.warn("Use species.reproduction_namespace for adjusted_fitness",
                      DeprecationWarning, stacklevel=2)
        self.reproduction_namespace.adjusted_fitness = None
    
    adjusted_fitness = property(_getAdjustedFitness,
                                _setAdjustedFitness,
                                _delAdjustedFitness)

    def _getFitness(self): # pragma: no cover
        """
        Backwards compatibility wrapper for species.fitness;
        use species.stagnation_namespace.fitness instead.
        """
        warnings.warn("Use species.stagnation_namespace for fitness",
                      DeprecationWarning, stacklevel=2)
        return self.stagnation_namespace.fitness

    def _setFitness(self,value): # pragma: no cover
        if not isinstance(value, float):
            raise TypeError(
                "Fitness ({0!r}) should be a float, not {1!s}".format(
                    value, type(value)))
        warnings.warn("Use species.stagnation_namespace for fitness",
                      DeprecationWarning, stacklevel=2)
        self.stagnation_namespace.fitness = value

    def _delFitness(self): # pragma: no cover
        warnings.warn("Use species.stagnation_namespace for fitness",
                      DeprecationWarning, stacklevel=2)
        self.stagnation_namespace.fitness = None

    fitness = property(_getFitness,
                       _setFitness,
                       _delFitness)

    def _getLastImproved(self): # pragma: no cover
        """
        Backwards compatibility wrapper for species.last_improved;
        use species.stagnation_namespace.last_improved instead.
        """
        warnings.warn("Use species.stagnation_namespace for last_improved",
                      DeprecationWarning, stacklevel=2)
        return self.stagnation_namespace.last_improved

    def _setLastImproved(self, value): # pragma: no cover
        if not isinstance(value, int):
            raise TypeError(
                "Last_improved ({0!r}) should be an int, not {1!s}".format(
                    value, type(value)))
        warnings.warn("Use species.stagnation_namespace for last_improved",
                      DeprecationWarning, stacklevel=2)
        self.stagnation_namespace.last_improved = value

    last_improved = property(_getLastImproved,
                             _setLastImproved)

    def _getFitnessHistory(self): # pragma: no cover
        """
        Partial (due to being a list) backwards compatibility
        wrapper for species.fitness_history; use
        species.stagnation_namespace.last_improved instead
        """
        warnings.warn("Use species.stagnation_namespace for fitness_history",
                      stacklevel=2)
        return self.stagnation_namespace.fitness_history

    fitness_history = property(_getFitnessHistory)

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
    """Encapsulates the default speciation scheme."""
    __desired_species_num_cache = {}

    def __init__(self, config, reporters, reproduction=None):
        # pylint: disable=super-init-not-called
        self.species_set_config = config
        self.reporters = reporters
        self.reproduction = reproduction
        self.indexer = count(1)
        self.species = {}
        self.genome_to_species = {}
        self.orig_compatibility_threshold = config.compatibility_threshold

        self.check_species_size_config(config)

        if ((config.compatibility_threshold_adjust.lower() != 'fixed') or
            (reproduction is not None)):
            self.min_pop_seen = 10**sys.float_info.dig
            self.max_pop_seen = 0
            if reproduction is None: # pragma: no cover
                raise RuntimeError(
                    "Need reproduction instance for species info (threshold_adjust {0!s})".format(
                        config.compatibility_threshold_adjust))
            self.threshold_adjust_dict = reproduction.get_species_size_info()
            if ((config.compatibility_threshold_adjust.lower() != 'fixed') or
                (self.threshold_adjust_dict['genome_config'] is not None)):
                if self.threshold_adjust_dict['genome_config'] is None: # pragma: no cover
                    raise RuntimeError(
                        "Need genome_config for species info (threshold_adjust {0!s})".format(
                            config.compatibility_threshold_adjust))
                self.threshold_adjust_dict.update(
                    self.threshold_adjust_dict['genome_config'].get_compatibility_info())
                # below is based on configuration values from Stanley's website as
                # compared to advised adjustment size.
                self.base_threshold_adjust = 0.3*max(
                    self.threshold_adjust_dict['weight_coefficient'],
                    (self.threshold_adjust_dict['disjoint_coefficient']/2),
                    (self.orig_compatibility_threshold/6))
            if config.compatibility_threshold_adjust.lower() == 'number':
                # for max_stagnation
                self.threshold_adjust_dict.update(
                    self.threshold_adjust_dict['stagnation'].get_stagnation_info())

    @staticmethod
    def check_species_size_config(config):
        if config.desired_species_num_max < config.desired_species_num_min:
            raise ValueError(
                "Desired_species_num_max {0:n} but num_min {1:n}".format(
                    config.desired_species_num_max, config.desired_species_num_min))

        if config.compatibility_threshold_adjust.lower() == 'choices':
            if not (0.0 <= config.desired_poss_species_pow <= 1.0):
                raise ValueError(
                    "Desired_poss_species_pow {0:n} invalid - must be between 0.0 and 1.0".format(
                        config.desired_poss_species_pow))

    @classmethod
    def parse_config(cls, param_dict):
        return DefaultClassConfig(param_dict,
                                  [ConfigParameter('compatibility_threshold', float),
                                   ConfigParameter('compatibility_threshold_adjust',
                                                   str, 'fixed'),
                                   ConfigParameter('desired_species_num_max',
                                                   int, 0, default_ok=True),
                                   ConfigParameter('desired_species_num_min',
                                                   int, 0, default_ok=True),
                                   ConfigParameter('desired_poss_species_pow',
                                                   float, 0.5, default_ok=True),
                                   ConfigParameter('spread_poss_species_range',
                                                   bool, False, default_ok=True)])

    def refresh_species_size_config(self, pop_size=None, reproduction=None, # DOCUMENT!
                                    genome_config=None, stagnation=None):
        config = self.species_set_config
        self.check_species_size_config(config)

        if config.compatibility_threshold_adjust.lower() != 'fixed':
            self.__desired_species_num_cache = {}
            if pop_size is not None:
                self.min_pop_seen = self.max_pop_seen = pop_size
            if reproduction is not None:
                self.reproduction = reproduction
            if self.reproduction is not None:
                self.threshold_adjust_dict = reproduction.get_species_size_info()
            else: # pragma: no cover
                raise RuntimeError("Need reproduction instance for species info")
            if genome_config is not None:
                self.threshold_adjust_dict['genome_config'] = genome_config
            if self.threshold_adjust_dict['genome_config'] is not None:
                self.threshold_adjust_dict.update(
                    self.threshold_adjust_dict['genome_config'].get_compatibility_info())
                # below is based on configuration values from Stanley's website as
                # compared to advised adjustment size.
                self.base_threshold_adjust = 0.3*max(
                    self.threshold_adjust_dict['weight_coefficient'],
                    (self.threshold_adjust_dict['disjoint_coefficient']/2.0),
                    (self.orig_compatibility_threshold/6.0))
            else: # pragma: no cover
                raise RuntimeError("Need genome_config for species info")
            if stagnation is not None:
                self.threshold_adjust_dict['stagnation'] = stagnation
            if self.threshold_adjust_dict['stagnation'] is not None:
                self.threshold_adjust_dict.update(
                    self.threshold_adjust_dict['stagnation'].get_stagnation_info())
            elif config.compatibility_threshold_adjust.lower() == 'number': # pragma: no cover
                raise RuntimeError("Need stagnation for species info")
        else:
            warnings.warn(
                "Calling refresh_species_size_config with" +
                " compatibility_threshold_adjust {0!r}".format(
                    config.compatibility_threshold_adjust) +
                " does not make sense")


    def _find_desired_num_species(self, pop_size_high, pop_size_low): # DOCUMENT!
        for_cache = tuple([pop_size_high,pop_size_low])
        if for_cache in self.__desired_species_num_cache:
            return self.__desired_species_num_cache[for_cache]
        config = self.species_set_config
        max_num_usable = math.floor(pop_size_high/self.threshold_adjust_dict['min_size'])
        if max_num_usable < 3:
            raise ValueError(
                "Pop_size {0:n} is too low for effective min species size {1:n}".format(
                    pop_size_high, self.threshold_adjust_dict['min_size']))

        poss_num_high = min(max_num_usable, # just in case
                            math.ceil(pop_size_high/
                                      self.threshold_adjust_dict['min_OK_size']))

        if config.desired_species_num_min > 2: # NEED TEST!
            max_use = max((poss_num_high-1),2)
            if config.desired_species_num_min > max_use: # NEED TEST!
                warnings.warn(
                    "Desired_species_num_min {0:n} is too high for pop_size {1:n};".format(
                        config.desired_species_num_min,pop_size_high)
                    + " adjusting to max {0:n}".format(max_use))
                config.desired_species_num_min = max_use
        elif ((config.desired_species_num_min != 0)
              and (config.desired_species_num_min != 2)): # NEED TEST!
            warnings.warn(
                "Desired_species_num_min of {0:n} not valid; treating as 0 (autoconfigure)".format(
                    config.desired_species_num_min))
            config.desired_species_num_min = 0

        if config.desired_species_num_max > 3:
            if config.desired_species_num_max > max_num_usable: # NEED TEST!
                warnings.warn(
                    "Desired_species_num_max {0:n} is too high for pop_size {1:n};".format(
                        config.desired_species_num_max,pop_size_high)
                    + " adjusting to max {0:n}".format(max_num_usable))
                config.desired_species_num_max = max_num_usable
        elif ((config.desired_species_num_max != 0)
              and (config.desired_species_num_max != 3)): # NEED TEST!
            warnings.warn(
                "Desired_species_num_max of {0:n} not valid; treating as 0 (autoconfigure)".format(
                    config.desired_species_num_max))
            config.desired_species_num_max = 0


        if config.desired_species_num_min:
            to_return_low = config.desired_species_num_min
        else:
            to_return_low = max(2,min((poss_num_high-1),
                                      math.ceil(pop_size_low/
                                                self.threshold_adjust_dict['min_good_size'])))

        if config.desired_species_num_max > to_return_low:
            to_return_high = config.desired_species_num_max
        elif config.desired_species_num_max: # NEED TEST!
            warnings.warn(
                    "Desired_species_num_max {0:n} is too low for min (autoconfigure?);".format(
                    config.desired_species_num_max)
                    + " adjusting to min+1 {0:n}".format(to_return_low+1))
            config.desired_species_num_max = to_return_low+1
            to_return_high = to_return_low+1
        elif poss_num_high < 3:
            raise ValueError(
                "Pop_size {0:n} is too low to determine desired num species".format(pop_size_high)
                + " given min_OK_size {1:n}".format(
                    self.threshold_adjust_dict['min_OK_size']))
        else:
            to_return_high = min(max_num_usable,max(poss_num_high,(to_return_low+1)))

        self.__desired_species_num_cache[for_cache] = (to_return_high, to_return_low)

        return (to_return_high,to_return_low)


    def _adjust_compatibility_threshold(self, increase, curr_tmean, max_rep_dist): # DOCUMENT!
        old_threshold = self.species_set_config.compatibility_threshold
        if increase:
            mult_threshold = 1.05*old_threshold
            add_threshold = old_threshold + self.base_threshold_adjust
            if old_threshold >= self.orig_compatibility_threshold: # cautious increase
                new_threshold = min(mult_threshold,add_threshold)
            else: # currently below original, can increase faster
                new_threshold = max(min(mult_threshold,add_threshold),
                                    min(self.orig_compatibility_threshold,
                                        max(mult_threshold,add_threshold)))
            if (new_threshold < curr_tmean < self.orig_compatibility_threshold):
                new_threshold = min(curr_tmean,((new_threshold
                                                 + self.orig_compatibility_threshold)/2))
            which = 'increased'
        else:
            div_threshold = old_threshold/1.05
            sub_threshold = old_threshold - self.base_threshold_adjust
            if old_threshold <= self.orig_compatibility_threshold: # cautious decrease
                new_threshold = max(div_threshold,sub_threshold)
            else: # currently above original, can decrease faster
                new_threshold = min(max(div_threshold,sub_threshold),
                                    max(self.orig_compatibility_threshold,
                                        min(div_threshold,sub_threshold)))

            new_threshold = max(new_threshold,NORM_EPSILON,min(max_rep_dist,old_threshold))
            if new_threshold >= old_threshold: # pragma: no cover
                return
            which = 'decreased'
        self.species_set_config.compatibility_threshold = new_threshold
        self.reporters.info(
            "Compatibility threshold (orig {0:n}) {1!s} by {2:n} to {3:n} (from {4:n})".format(
                self.orig_compatibility_threshold,
                which,
                abs(new_threshold-old_threshold),
                new_threshold,
                old_threshold))

    def choose_species(self, config, genome, gid, distances, candidates):
        """
        Decide which species a genome should be in, out of those sufficiently genetically similar.

        This method is meant to be replaced by classes subclassing DefaultSpeciesSet,
        such as to additionally consider behavioral distances. This default version
        simply chooses the minimum genetic distance.
        """
        ignored_sdist, sid, ignored_rep = min(candidates, key=lambda x: x[0])
        return sid

    def decide_on_threshold(self, num_species,
                            desired_num_species_high,
                            desired_num_species_low,
                            gdtmean,
                            max_rep_dist,
                            generation,
                            num_choices):
        """Replaceable speciation threshold-determination methods."""
        if self.species_set_config.compatibility_threshold_adjust.lower() == 'number':
            if num_species < desired_num_species_low:
                self.reporters.info(
                    "Species num {0:n} below desired minimum {1:n}".format(
                        num_species, desired_num_species_low))
                self._adjust_compatibility_threshold(increase=False,
                                                     curr_tmean=gdtmean,
                                                     max_rep_dist=max_rep_dist)
            elif num_species > desired_num_species_high:
                self.reporters.info(
                        "Species num {0:n} above desired maximum {1:n}".format(
                        num_species, desired_num_species_high))
                if (('max_stagnation' not in self.threshold_adjust_dict)
                    or (generation >= self.threshold_adjust_dict['max_stagnation'])
                    or (self.species_set_config.compatibility_threshold <
                        self.orig_compatibility_threshold)):
                    self._adjust_compatibility_threshold(increase=True,
                                                         curr_tmean=gdtmean,
                                                         max_rep_dist=max_rep_dist)
        elif self.species_set_config.compatibility_threshold_adjust.lower() == 'choices':

            mean_desired_num_species = (desired_num_species_high+desired_num_species_low)/2.0
            base_min_desired_choices = max(2,math.floor(
                math.pow(mean_desired_num_species,
                         self.species_set_config.desired_poss_species_pow)))
            base_max_desired_choices = min((desired_num_species_high-1),math.ceil(
                math.pow(mean_desired_num_species,
                         self.species_set_config.desired_poss_species_pow)))
            if self.species_set_config.spread_poss_species_range:
                min_desired_choices = max(2,math.floor(
                    math.pow(desired_num_species_low,
                             self.species_set_config.desired_poss_species_pow)))
                if num_species > desired_num_species_high:
                    min_desired_choices *= num_species/desired_species_num_high
                    min_desired_choices = min(base_min_desired_choices,
                                              math.ceil(min_desired_choices))
                max_desired_choices = min((desired_num_species_high-1),math.ceil(
                    math.pow(desired_num_species_high,
                             self.species_set_config.desired_poss_species_pow)))
                if num_species < desired_num_species_low:
                    max_desired_choices *= num_species/desired_num_species_low
                    max_desired_choices = max(base_max_desired_choices,
                                              math.floor(max_desired_choices))
            else:
                min_desired_choices = base_min_desired_choices
                max_desired_choices = base_max_desired_choices
            if min_desired_choices > max_desired_choices: # pragma: no cover
                tmp_min = max(2,(max_desired_choices-1))
                tmp_max = min((desired_num_species_high-1),(max(tmp_min,min_desired_choices)+1))
                if tmp_min > tmp_max:
                    raise ValueError(
                        "Getting min_desired_choices {0:n}, tmp_min {1:n},".format(
                            min_desired_choices, tmp_min) +
                        " max_desired_choices {0:n}, tmp_max {1:n}".format(
                            max_desired_choices, tmp_max))
                min_desired_choices = tmp_min
                max_desired_choices = tmp_max

            num_low_choices = sum([1 for x in num_choices if x < min_desired_choices])
            num_high_choices = sum([1 for x in num_choices if x > max_desired_choices])
            num_ok_choices = len(num_choices) - num_low_choices - num_high_choices
            if ((num_species > desired_num_species_low) and
                (num_low_choices > max(num_high_choices, num_ok_choices))):
                # may increase threshold (eventually decreases the total # of species)
                do_increase = False
                if num_species >= desired_num_species_high:
                    do_increase = True
                elif num_species >= math.ceil(mean_desired_num_species):
                    diff1 = desired_num_species_high - math.ceil(mean_desired_num_species)
                    diff2 = num_species - math.ceil(mean_desired_num_species)
                    threshold = 0.5 - ((0.5-(1/3))*(diff2/max(1.0,diff1)))
                    if num_low_choices >= (len(num_choices)*threshold):
                        do_increase = True
                elif num_low_choices > (len(num_choices)*0.5):
                    diff1 = math.floor(mean_desired_num_species) - (desired_num_species_low+1)
                    diff2 = num_species - (desired_num_species_low+1)
                    threshold2 = 2.0 - (diff2/max(1.0,diff1))
                    if num_low_choices >= (num_high_choices*threshold2):
                        do_increase = True
                if do_increase:
                    self.reporters.info(
                        "Species num {0:n} above desired minimum {1:n}".format(
                            num_species, desired_num_species_low) +
                        " and too many ({0:n}) with species choices below {1:n}".format(
                            num_low_choices, min_desired_choices))
                    self._adjust_compatibility_threshold(increase=True,
                                                         curr_tmean=gdtmean,
                                                         max_rep_dist=max_rep_dist)
            elif ((num_species < desired_num_species_high) and
                  (num_high_choices > max(num_low_choices, num_ok_choices))):
                # may decrease threshold (eventually increases the total # of species)
                do_decrease = False
                if num_species <= desired_num_species_low:
                    do_decrease = True
                elif num_species <= math.floor(mean_desired_num_species):
                    diff1 = math.floor(mean_desired_num_species) - desired_num_species_low
                    diff2 = num_species - desired_num_species_low
                    threshold = 0.5 - ((0.5-(1/3))*(diff2/max(1.0,diff1)))
                    if num_high_choices >= (len(num_choices)*threshold):
                        do_increase = True
                elif num_high_choices > (len(num_choices)*0.5):
                    diff1 = (desired_num_species_high-1) - math.ceil(mean_desired_num_species)
                    diff2 = (desired_num_species_high-1) - num_species
                    threshold2 = 2.0 - (diff2/max(1.0,diff1))
                    if num_high_choices >= (num_low_choices*threshold2):
                        do_decrease = True
                if do_decrease:
                    self.reporters.info(
                        "Species num {0:n} below desired maximum {1:n}".format(
                            num_species, desired_num_species_high) +
                        " and too many ({0:n}) with species choices above {1:n}".format(
                            num_high_choices, max_desired_choices))
                    self._adjust_compatibility_threshold(increase=False,
                                                         curr_tmean=gdtmean,
                                                         max_rep_dist=max_rep_dist)
        elif self.species_set_config.compatibility_threshold_adjust.lower() != 'fixed':
            raise ValueError(
                "Unknown compatibility_threshold_adjust {!r}".format(
                    self.species_set_config.compatibility_threshold_adjust))

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

        num_choices = []
        # Partition population into species based on genetic similarity.
        for gid in unspeciated_list:
            g = population[gid]

            # Find the species with the most similar representative.
            candidates = []
            for sid, rid in iteritems(new_representatives):
                rep = population[rid]
                d = distances(rep, g)
                if d < self.species_set_config.compatibility_threshold:
                    candidates.append((d, sid, rep))

            num_choices.append(len(candidates))
            if candidates:
                sid = self.choose_species(config, gid, g, distances, candidates)
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

        if self.species_set_config.compatibility_threshold_adjust.lower() != 'fixed':
            if generation < 1:
                self.reporters.info(
                    "Min_size is {0:n}, min_OK_size is {1:n}, min_good_size is {2:n}".format(
                        self.threshold_adjust_dict['min_size'],
                        self.threshold_adjust_dict['min_OK_size'],
                        self.threshold_adjust_dict['min_good_size']))
            if ((self.min_pop_seen <= self.max_pop_seen) and
                ((self.min_pop_seen > (len(population)*2)) or
                 (self.max_pop_seen < (len(population)/2.0)))): # pragma: no cover
                self.min_pop_seen = self.max_pop_seen = len(population)
            else:
                self.min_pop_seen = min(self.min_pop_seen,len(population))
                self.max_pop_seen = max(self.max_pop_seen,len(population))
            desired_num_species_high, desired_num_species_low = self._find_desired_num_species(
                self.max_pop_seen, self.min_pop_seen)
            if ((self.species_set_config.compatibility_threshold <= max_rep_dist)
                and (len(self.species) > desired_num_species_low)): # pragma: no cover
                self._adjust_compatibility_threshold(increase=True,
                                                     curr_tmean=gdtmean,
                                                     max_rep_dist=max_rep_dist)
            else:
                self.decide_on_threshold(num_species=len(self.species),
                                         desired_num_species_high=desired_num_species_high,
                                         desired_num_species_low=desired_num_species_low,
                                         gdtmean=gdtmean,
                                         max_rep_dist=max_rep_dist,
                                         generation=generation,
                                         num_choices=num_choices)

    def get_species_id(self, individual_id):
        return self.genome_to_species[individual_id]

    def get_species(self, individual_id): # NEED TEST!
        sid = self.genome_to_species[individual_id]
        return self.species[sid]
