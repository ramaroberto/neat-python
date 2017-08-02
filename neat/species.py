"""Divides the population into species based on genomic distances."""
from itertools import count

from neat.math_util import mean, stdev
from neat.six_util import iteritems, iterkeys, itervalues
from neat.config import ConfigParameter
from neat.mypy_util import * # pylint: disable=unused-wildcard-import

if MYPY: # pragma: no cover
    from neat.mypy_util import DefaultGenome, DefaultGenomeConfig, DefaultClassConfig, Config # pylint: disable=unused-import
    from neat.reporting import ReporterSet # pylint: disable=unused-import
else:
    from neat.config import DefaultClassConfig

class Species(object):
    def __init__(self,
                 key, # type: SpeciesKey
                 generation # type: int
                 ):
        # type: (...) -> None
        self.key = key
        self.created = generation
        self.last_improved = generation
        self.representative = None # type: Optional[KnownGenome] # XXX
        self.members = {} # type: Dict[GenomeKey, KnownGenome] # XXX
        self.fitness = None # type: Optional[float]
        self.adjusted_fitness = None # type: Optional[float]
        self.fitness_history = [] # type: List[float]

    def update(self,
               representative, # type: KnownGenome # XXX
               members # type: Dict[GenomeKey, KnownGenome] # XXX
               ):
        # type: (...) -> None
        self.representative = representative
        self.members = members

    def get_fitnesses(self): # type: () -> List[Optional[float]]
        return [m.fitness for m in itervalues(self.members)]


class GenomeDistanceCache(object):
    def __init__(self, config): # type: (DefaultGenomeConfig) -> None
        self.distances = {} # type: Dict[Tuple[GenomeKey, GenomeKey], float]
        self.config = config
        self.hits = 0 # type: int # c_type: c_uint
        self.misses = 0 # type: int # c_type: c_uint

    def __call__(self, genome0, genome1): # type: (KnownGenome, KnownGenome) -> float # XXX
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

    def __init__(self, config, reporters): # type: (DefaultClassConfig, ReporterSet) -> None
        # pylint: disable=super-init-not-called
        self.species_set_config = config
        self.reporters = reporters
        self.indexer = count(1)
        self.species = {} # type: Dict[SpeciesKey, Species]
        self.genome_to_species = {} # type: Dict[GenomeKey, SpeciesKey]

    @classmethod
    def parse_config(cls, param_dict): # type: (Dict[str, str]) -> DefaultClassConfig
        return DefaultClassConfig(param_dict,
                                  [ConfigParameter('compatibility_threshold', float)])

    def speciate(self,
                 config, # type: Config # CORRECTION
                 population, # type: Dict[GenomeKey, KnownGenome] # XXX
                 generation # type: int
                 ):
        # type: (...) -> None
        """
        Place genomes into species by genetic similarity.

        Note that this method assumes the current representatives of the species are from the old
        generation, and that after speciation has been performed, the old representatives should be
        dropped and replaced with representatives from the new generation.  If you violate this
        assumption, you should make sure other necessary parts of the code are updated to reflect
        the new behavior.
        """
        assert isinstance(population, dict)

        compatibility_threshold = self.species_set_config.compatibility_threshold # type: ignore
        compatibility_threshold = cast(float, compatibility_threshold)

        # Find the best representatives for each existing species.
        unspeciated = set(iterkeys(population)) # type: Set[GenomeKey]
        distances = GenomeDistanceCache(config.genome_config)
        new_representatives = {}
        new_members = {}
        for sid, s in iteritems(self.species):
            candidates = [] # type: List[Tuple[float, KnownGenome]] # XXX
            for gid in unspeciated:
                g = population[gid]
                d = distances(s.representative, g)
                candidates.append((d, g))

            # The new representative is the genome closest to the current representative.
            ignored_rdist, new_rep = min(candidates, key=lambda x: x[0])
            new_rid = cast(GenomeKey,new_rep.key)
            new_representatives[sid] = new_rid
            new_members[sid] = [new_rid]
            unspeciated.remove(new_rid)

        # Partition population into species based on genetic similarity.
        while unspeciated:
            gid = unspeciated.pop()
            g = population[gid]

            # Find the species with the most similar representative.
            candidates = []
            for sid, rid in iteritems(new_representatives):
                rep = population[rid]
                d = distances(rep, g)
                if d < compatibility_threshold:
                    candidates.append((d, sid))

            if candidates:
                ignored_sdist, sid = min(candidates, key=lambda x: x[0])
                new_members[sid].append(gid)
            else:
                # No species is similar enough, create a new species, using
                # this genome as its representative.
                sid = cast(SpeciesKey,next(self.indexer))
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
        gdstdev = stdev(itervalues(distances.distances))
        self.reporters.info(
            'Mean genetic distance {0:.3f}, standard deviation {1:.3f}'.format(gdmean, gdstdev))

    def get_species_id(self, individual_id): # type: (GenomeKey) -> SpeciesKey
        return self.genome_to_species[individual_id]

    def get_species(self, individual_id): # type: (GenomeKey) -> Species
        sid = self.genome_to_species[individual_id]
        return self.species[sid]
