"""
Has the built-in aggregation functions, methods for using them,
and methods for adding new user-defined ones.
"""
from __future__ import division

import math
import sys
import warnings

from operator import mul

from neat.multiparameter import MultiParameterSet
from neat.multiparameter import BadFunctionError as InvalidAggregationFunction # pylint: disable=unused-import
from neat.math_util import mean, median2

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

def sum_mean_aggregation(x, a):
    assert a <= 1.0
    assert a >= 0.0

    input_list = list(map(float,x))

    num_input = len(input_list)

    if num_input == 1:
        return input_list[0]

    mult = (1+(a*(num_input - 1)))/num_input

    return sum(input_list)*mult

def product_mean_aggregation(x, a, use_median):
    assert a <= 1.0
    assert a >= 0.0

    input_list = list(map(float,x))

    num_input = len(input_list)

    if num_input == 1:
        return input_list[0]

    power = (1+(a*(num_input - 1)))/num_input

    tmp_product = product_aggregation(input_list)

    if use_median:
        return math.copysign(math.pow(abs(tmp_product), power), median2(input_list))
    
    return math.copysign(math.pow(abs(tmp_product), power), tmp_product)


def sum_product_aggregation(x, a):
    assert a <= 1.0
    assert a >= 0.0

    return ((1.0-a)*product_aggregation(x))+(a*sum(x))

def sum_product_mean_aggregation(x, a, b, use_median):
    assert a <= 1.0
    assert a >= 0.0
    assert b <= 1.0
    assert b >= 0.0

    return ((b*sum_mean_aggregation(x, a))+
            ((1.0-b)*product_mean_aggregation(x, a, use_median)))
    

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
        self.add('sum_mean', sum_mean_aggregation,
                 a={'min_value':0.0, 'max_value':1.0})
        self.add('product_mean', product_mean_aggregation,
                 a={'min_value':0.0, 'max_value':1.0},
                 use_median={'param_type': 'bool'})
        self.add('sum_product', sum_product_aggregation,
                 a={'min_value':0.0, 'max_value':1.0})
        self.add('sum_product_mean', sum_product_mean_aggregation,
                 a={'min_value':0.0, 'max_value':1.0},
                 b={'min_value':0.0, 'max_value':1.0},
                 use_median={'param_type': 'bool'})

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

