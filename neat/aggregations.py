"""
Has the built-in aggregation functions, methods for using them,
and methods for adding new user-defined ones.
"""
import sys
import warnings

from operator import mul

from neat.multiparameter import MultiParameterSet
from neat.multiparameter import BadFunctionError as InvalidAggregationFunction # pylint: disable=unused-import
from neat.math_util import mean, median2

from neat.mypy_util import * # pylint: disable=unused-wildcard-import

if MYPY: # pragma: no cover
    from neat.multiparameter import (MultiParameterFunctionInstance, # pylint: disable=unused-import
                                     MPAgFunc, NormAgFunc)
    AgFunc = Union[MPAgFunc, NormAgFunc]

if sys.version_info[0] > 2:
    from functools import reduce

def product_aggregation(x): # type: (Iterable[float]) -> float
    return reduce(mul, x, 1.0)

def sum_aggregation(x): # type: (Iterable[float]) -> float
    return sum(x)

def max_aggregation(x): # type: (Iterable[float]) -> float
    return max(x)

def min_aggregation(x): # type: (Iterable[float]) -> float
    return min(x)

def maxabs_aggregation(x): # type: (Iterable[float]) -> float
    return max(x, key=abs)

def median_aggregation(x): # type: (Iterable[float]) -> float
    return median2(x)

def mean_aggregation(x): # type: (Iterable[float]) -> float
    return mean(x)

def max_median_min_aggregation(x, # type: Iterable[float]
                               a # type: float
                               ):
    # type: (...) -> float
    assert a <= 1.0
    assert a >= -1.0
    median_weight = 1.0-abs(a)

    if a > 0.0:
        return (((1.0-median_weight)*max(x))+
                (median_weight*median2(x)))
    if a < 0.0:
        return (((1.0-median_weight)*min(x))+
                (median_weight*median2(x)))
    return median2(x)

def maxabs_mean_aggregation(x, # type: Iterable[float]
                            a # type: float
                            ):
    # type: (...) -> float
    assert a <= 1.0
    assert a >= 0.0
    return ((1.0-a)*mean(x))+(a*maxabs_aggregation(x))

class AggregationFunctionSet(object):
    """Contains aggregation functions and methods to add and retrieve them."""
    def __init__(self,
                 multiparameterset=None # type: Optional[MultiParameterSet]
                 ):
        # type: (...) -> None
        if multiparameterset is None:
            warn_string = ("Aggregation init called without multiparameterset:" +
                           " may cause multiple instances of it")
            multiparameterset = MultiParameterSet('aggregation')
            warnings.warn(warn_string)
        self.multiparameterset = multiparameterset
        self.add('product', product_aggregation)
        self.add('sum', sum_aggregation)
        self.add('max', max_aggregation)
        self.add('min', min_aggregation)
        self.add('maxabs', maxabs_aggregation)
        self.add('median', median_aggregation)
        self.add('mean', mean_aggregation)
        self.add('max_median_min', max_median_min_aggregation,
                 a={'min_value':-1.0, 'max_value':1.0})
        self.add('maxabs_mean', maxabs_mean_aggregation,
                 a={'min_value':0.0, 'max_value':1.0})


    def add(self,
            name, # type: str
            function, # type: AgFunc
            **kwargs # type: Dict[str, Union[str, float]]
            ):
        # type: (...) -> None
        self.multiparameterset.add_func(name, function, 'aggregation', **kwargs)

    def get(self, name): # type: (Union[str, MultiParameterFunctionInstance]) -> AgFunc
        to_return = self.multiparameterset.get_func(name, 'aggregation')
        if MYPY: # pragma: no cover
            to_return = cast(AgFunc, to_return)
        return to_return

    def __getitem__(self, index): # type: (Union[str, MultiParameterFunctionInstance]) -> AgFunc
        warnings.warn("Use get, not indexing ([{!r}]), for aggregation functions".format(index),
                      DeprecationWarning)
        return self.get(index)

    def is_valid(self,name): # type: (str) -> bool
        return self.multiparameterset.is_valid_func(name, 'aggregation')
