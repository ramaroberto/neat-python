"""Utilities for working with mypy for type checking."""

MYPY = False # pragma: no cover

__all__ = ['cast', 'NodeKey', 'GenomeKey', 'SpeciesKey', 'ConnKey', 'GeneKey', 'MYPY']

if MYPY: # pragma: no cover
    import sys
    if sys.version_info[0] >= 3:
        from typing import TextIO # pylint: disable=unused-import
    else:
        from typing import IO as TextIO # pylint: disable=unused-import
    from typing import (Iterable, Set, List, Sequence, NewType, # pylint: disable=unused-import
                        Tuple, Optional, Union, cast, Dict, Any, Callable)
    from neat.config import DefaultClassConfig, Config # pylint: disable=unused-import
    from neat.genome import DefaultGenomeConfig, DefaultGenome # pylint: disable=unused-import
    from neat.iznn import IZGenome # pylint: disable=unused-import
    KnownConfig = Union[DefaultClassConfig, Config, DefaultGenomeConfig] # XXX
    KnownGenome = Union[DefaultGenome, IZGenome]  # XXX
    NodeKey = NewType('NodeKey', int)
    ConnKey = Tuple[NodeKey, NodeKey]
    GeneKey = Union[NodeKey, ConnKey]
    GenomeKey = NewType('GenomeKey', int) # c_type: c_uint
    SpeciesKey = NewType('SpeciesKey', int) # c_type: c_uint


    __all__ += ['Iterable', 'Set', 'List', 'Sequence', # not NewType
                'Tuple', 'Optional', 'Union', 'Dict', 'Any', 'TextIO',
                'KnownConfig', 'KnownGenome']
else:
    NodeKey = int # pylint: disable=invalid-name
    ConnKey = tuple # pylint: disable=invalid-name
    GenomeKey = 'uint' # pylint: disable=invalid-name
    SpeciesKey = 'uint' # pylint: disable=invalid-name
    GeneKey = (int, tuple) # pylint: disable=invalid-name

    def cast(desired_type, var):
        if desired_type is None:
            return var
        if desired_type == 'uint':
            assert isinstance(var, int)
            assert var >= 0
            return var
        try:
            assert isinstance(var, desired_type)
        except TypeError:
            return var
        else:
            return var
