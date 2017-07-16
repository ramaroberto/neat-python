"""
Has the built-in aggregation functions, methods for using them,
and methods for adding new user-defined ones.
"""
import sys
import warnings

from neat.math_util import mean, median2

from operator import mul

from neat.multiparameter import MultiParameterSet

if sys.version_info[0] > 2:
    from functools import reduce

def product_aggregation(x): # note: `x` is a list or other iterable
    return reduce(mul, x, 1.0)

def sum_aggregation(x):
    return sum(x)

def max_aggregation(x):
    return max(x)

def min_aggregation(x):
    return min(x)

def maxabs_aggregation(x):
    return max(x, key=abs)

def median_aggregation(x):
    return median2(x)

def mean_aggregation(x):
    return mean(x)

def max_median_min_aggregation(x, a):
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

def maxabs_mean_aggregation(x, a):
    assert a <= 1.0
    assert a >= 0.0
    return ((1.0-a)*mean(x))+(a*maxabs_aggregation(x))

class InvalidAggregationFunction(TypeError):
    pass


def validate_aggregation(function): # TODO: Recognize when need `reduce`
    if not isinstance(function,
                      (types.BuiltinFunctionType,
                       types.FunctionType,
                       types.LambdaType)):
        raise InvalidAggregationFunction("A function object is required.")

    if not (function.__code__.co_argcount >= 1):
        raise InvalidAggregationFunction("A function taking at least one argument is required")


class AggregationFunctionSet(object):
    """Contains aggregation functions and methods to add and retrieve them."""
    
    def __init__(self, multiparameterset=None):
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

    def add(self, name, function, **kwargs):
        self.multiparameterset.add_func(name, function, 'aggregation', **kwargs)

    def get(self, name):
        return self.multiparameterset.get_func(name, 'aggregation')

    def __getitem__(self, index):
        warnings.warn("Use get, not indexing ([{!r}]), for aggregation functions".format(index),
                      DeprecationWarning)
        return self.get(index)

    def is_valid(self, name):
        return self.multiparameterset.is_valid_func(name, 'aggregation')

