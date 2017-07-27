"""Utilities for working with mypy for type checking."""

MYPY = False

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
    import functools
    from types import FunctionType, LambdaType, BuiltinFunctionType
    import warnings

    AgFunc = (FunctionType, LambdaType, BuiltinFunctionType, functools.partial)
    ActFunc = AgFunc
    NormActFunc = ActFunc
    NormAgFunc = AgFunc
    MPActFunc = (FunctionType, LambdaType, functools.partial)
    MPAgFunc = (FunctionType, LambdaType, functools.partial)

    NodeKey = int # pylint: disable=invalid-name
    ConnKey = 'ConnKey' # pylint: disable=invalid-name
    GenomeKey = 'uint' # pylint: disable=invalid-name
    SpeciesKey = 'uint' # pylint: disable=invalid-name
    GeneKey = (int, tuple) # pylint: disable=invalid-name

    __all__ += ['AgFunc', 'ActFunc', 'NormActFunc', 'NormAgFunc',
                'MPActFunc', 'MPAgFunc']

    def cast(desired_type, var):
        if desired_type is None: # pragma: no cover
            return var
        if desired_type == 'uint':
            assert isinstance(var, int), "Var {0!r}, type '{1!s}', is not an int".format(var, type(var))
            assert var >= 0, "Var has negative value {:n}".format(var)
            return var
        if desired_type == 'ConnKey':
            assert isinstance(var, tuple), "Var {0!r}, type '{1!s}', is not a tuple".format(var, type(var))
            assert len(var) == 2
            assert isinstance(var[0], int)
            assert isinstance(var[1], int)
            assert var[1] >= 0
            return var
        try:
            assert isinstance(var,
                              desired_type), "Var {0!r}, type '{1!s}' is not desired type {2!r}".format(var,
                                                                                                        type(var),
                                                                                                        desired_type)
        except TypeError:
            warnings.warn("Desired_type {!r} not usable by isinstance".format(desired_type))
            return var
        else:
            return var
