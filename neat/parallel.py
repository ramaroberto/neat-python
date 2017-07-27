"""
Runs evaluation functions in parallel subprocesses,
in order to evaluate multiple genomes at once.
"""
from multiprocessing import Pool

from neat.mypy_util import * # pylint: disable=unused-wildcard-import

if MYPY: # pragma: no cover
    from typing import Callable # pylint: disable=unused-import

class ParallelEvaluator(object):
    def __init__(self,
                 num_workers, # type: int
                 eval_function, # type: Callable[[Tuple[KnownGenome, KnownConfig]], float]
                 timeout=None # type: Optional[float]
                 ):
        # type: (...) -> None
        """
        eval_function should take one argument, a tuple of
        (genome object, config object), and return
        a single float (the genome's fitness).
        """
        self.num_workers = num_workers
        self.eval_function = eval_function
        self.timeout = timeout
        self.pool = Pool(num_workers)

    def __del__(self): # type: () -> None
        self.pool.close() # should this be terminate?
        self.pool.join()

    def evaluate(self,
                 genomes, # type: List[Tuple[GenomeKey, KnownGenome]] # DOCS CORRECTION!
                 config # type: KnownConfig
                 ):
        # type: (...) -> None
        jobs = []
        for ignored_genome_id, genome in genomes:
            jobs.append(self.pool.apply_async(self.eval_function, (genome, config)))

        # assign the fitness back to each genome
        for job, (ignored_genome_id, genome) in zip(jobs, genomes):
            genome.fitness = job.get(timeout=self.timeout)
