"""Uses `pickle` to save and restore populations (and other aspects of the simulation state)."""
from __future__ import print_function

import gzip
import pickle
import random
import time

from neat.population import Population
from neat.reporting import BaseReporter

from neat.mypy_util import * # pylint: disable=unused-wildcard-import

if MYPY:
    from neat.species import DefaultSpeciesSet # pylint: disable=unused-import
    from neat.config import Config # pylint: disable=unused-import

class Checkpointer(BaseReporter):
    """
    A reporter class that performs checkpointing using `pickle`
    to save and restore populations (and other aspects of the simulation state).
    """
    def __init__(self,
                 generation_interval=100, # type: Optional[int]
                 time_interval_seconds=300 # type: Optional[float]
                 ):
        # type: (...) -> None
        """
        Saves the current state (at the end of a generation) every ``generation_interval`` generations or
        ``time_interval_seconds``, whichever happens first.

        :param generation_interval: If not None, maximum number of generations between save intervals
        :type generation_interval: int or None
        :param time_interval_seconds: If not None, maximum number of seconds between checkpoint attempts
        :type time_interval_seconds: float or None
        """
        self.generation_interval = generation_interval
        self.time_interval_seconds = time_interval_seconds

        self.current_generation = None # type: Optional[int]
        self.last_generation_checkpoint = -1 # type: int
        self.last_time_checkpoint = time.time() # type: float

    def start_generation(self, generation): # type: (int) -> None
        self.current_generation = generation

    def end_generation(self,
                       config, # type: Config
                       population, # type: Dict[GenomeKey, KnownGenome] # XXX
                       species_set # type: DefaultSpeciesSet # XXX
                       ):
        # type: (...) -> None
        checkpoint_due = False # type: bool

        if self.time_interval_seconds is not None:
            dt = time.time() - self.last_time_checkpoint
            if dt >= self.time_interval_seconds:
                checkpoint_due = True

        if (checkpoint_due is False) and (self.generation_interval is not None):
            dg = self.current_generation - self.last_generation_checkpoint
            if dg >= self.generation_interval:
                checkpoint_due = True

        if checkpoint_due:
            self.save_checkpoint(config, population, species_set, self.current_generation)
            self.last_generation_checkpoint = self.current_generation
            self.last_time_checkpoint = time.time()

    @staticmethod
    def save_checkpoint(config, # type: Config
                        population, # type: Dict[GenomeKey, KnownGenome] # XXX
                        species_set, # type: DefaultSpeciesSet # XXX
                        generation # type: int
                        ):
        # type: (...) -> None
        """ Save the current simulation state. """
        filename = 'neat-checkpoint-{0}'.format(generation) # type: str
        print("Saving checkpoint to {0}".format(filename))

        with gzip.open(filename, 'w', compresslevel=5) as f:
            data = (generation, config, population, species_set, random.getstate())
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def restore_checkpoint(filename): # type: (str) -> Population
        """Resumes the simulation from a previous saved point."""
        with gzip.open(filename) as f:
            (generation,
             config,
             population,
             species_set,
             rndstate
             ) = pickle.load(f) # type: int, Config, Dict[GenomeKey, KnownGenome], DefaultSpeciesSet, object
            random.setstate(rndstate)
            return Population(config, (population, species_set, generation))
