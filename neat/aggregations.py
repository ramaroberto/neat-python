"""
Has the built-in aggregation functions, methods for using them,
and methods for adding new user-defined ones.
"""
from __future__ import division

import math
import sys
import warnings

from operator import mul
from pprint import saferepr

from neat.multiparameter import MultiParameterSet
from neat.multiparameter import BadFunctionError as InvalidAggregationFunction # pylint: disable=unused-import
from neat.math_util import mean, median2, tmean, check_value_range

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

def tmean_aggregation(x):
    return tmean(x,trim=0.25)

def _check_value_range(a, min_val, max_val, caller, var_name): # TEST NEEDED!
    check_value_range(var=a,
                      min_val=min_val,
                      max_val=max_val,
                      caller=caller,
                      var_name=var_name,
                      add_name="aggregation")

def maxabs_mean_aggregation(x, a):
    _check_value_range(a, 0.0, 1.0, 'maxabs_mean', 'a')
    return ((1.0-a)*mean(x))+(a*maxabs_aggregation(x))

def multiparam_tmean_aggregation(x, a):
    _check_value_range(a, 0.0, 0.5, 'multiparam_tmean', 'a')
    return tmean(x,trim=a)

def maxabs_tmean_aggregation(x, extreme):
    _check_value_range(extreme, -1.0, 1.0, 'maxabs_tmean', 'extreme')
    if extreme >= 0.0:
        return maxabs_mean_aggregation(x, a=extreme)
    return multiparam_tmean_aggregation(x, a=abs(extreme/2))

def sum_product_aggregation(x, add_mult):
    _check_value_range(add_mult, 0.0, 1.0, 'sum_product', 'add_mult')
    return ((1.0-add_mult)*product_aggregation(x))+(add_mult*sum(x))

def max_median_min_aggregation(x, a):
    """Approximates percentiles."""
    _check_value_range(a, -1.0, 1.0, 'max_median_min', 'a')

    input_list = list(map(float,x))

    if len(input_list) == 1:
        return input_list[0]

    if a == 0.0:
        return median2(input_list)
    if a == 1.0:
        return max(input_list)
    if a == -1.0:
        return min(input_list)

    if a > 0.0:
        if len(input_list) > 3:
            # python sort avoids doing unnecessary work for already-sorted input
            input_list.sort()
            if (len(input_list) % 2) == 1:
                return max_median_min_aggregation(input_list[(len(input_list)//2):],
                                                  ((a*2.0)-1.0))
            return max_median_min_aggregation(input_list[((len(input_list)//2)+1):],
                                              ((a*2.0)-1.0))
        else:
            median_weight = 1.0-abs(a)
            return (((1.0-median_weight)*max(input_list))+
                    (median_weight*median2(input_list)))

    if len(input_list) > 3:
        input_list.sort()
        if (len(input_list) % 2) == 1:
            return max_median_min_aggregation(input_list[0:((len(input_list)//2)+1)],
                                              ((a*2.0)+1.0))
        return max_median_min_aggregation(input_list[0:(len(input_list)//2)],
                                          ((a*2.0)+1.0))

    median_weight = 1.0-abs(a)
    return (((1.0-median_weight)*min(input_list))+
            (median_weight*median2(input_list)))

def sum_mean_aggregation(x, average):
    _check_value_range(average, 0.0, 1.0, 'sum_mean', 'average')

    input_list = list(map(float,x))

    num_input = len(input_list)

    if num_input == 1:
        return input_list[0]

    a = 1.0-average

    mult = (1+(a*(num_input - 1)))/num_input

    return sum(input_list)*mult

def sum_maxabs_tmean_aggregation(x, average, extreme):
    _check_value_range(extreme, -1.0, 1.0, 'sum_maxabs_tmean', 'extreme')
    _check_value_range(average, 0.0, 1.0, 'sum_maxabs_tmean', 'average')

    input_list = list(map(float,x))

    num_input = len(input_list)

    if num_input == 1:
        return input_list[0]

    tmp_result = maxabs_tmean_aggregation(input_list, extreme)

    a = 1.0-average

    mult = (1+(a*(num_input - 1)))

    return tmp_result*mult


def product_mean_aggregation(x, average, use_median_sign):
    """Finds a compromise between a product and a geometric mean."""
    _check_value_range(average, 0.0, 1.0, 'product_mean', 'average')
    if not isinstance(use_median_sign, bool):
        raise TypeError(
            "Type of use_median_sign {0} must be bool, not {1}".format(
                saferepr(use_median_sign),type(use_median_sign)))

    input_list = list(map(float,x))

    num_input = len(input_list)

    if num_input == 1:
        return input_list[0]

    tmp_product = product_aggregation(input_list)
    if tmp_product == 0.0:
        return tmp_product

    a = 1.0-average

    power = (1+(a*(num_input - 1)))/num_input
    transformed_product = math.pow(abs(tmp_product), power)

    if use_median_sign:
        median2_result = median2(input_list)
        if median2_result:
            return math.copysign(transformed_product, median2_result)
    return math.copysign(transformed_product, tmp_product)

def sum_product_mean_aggregation(x, average, add_mult, use_median_sign):
    _check_value_range(average, 0.0, 1.0, 'sum_product_mean', 'average')
    _check_value_range(add_mult, 0.0, 1.0, 'sum_product_mean', 'add_mult')
    return ((add_mult*sum_mean_aggregation(x, average))+
            ((1.0-add_mult)*product_mean_aggregation(x, average, use_median_sign)))


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
        self.add('tmean', tmean_aggregation)

        self.add_shared_name('use_median_sign',
                             **{'param_type': 'bool'})
        self.add_shared_name('add_mult',
                             **{'min_value':0.0, 'max_value':1.0})
        self.add_shared_name('average',
                             **{'min_value':0.0, 'max_value':1.0})
        self.add_shared_name('extreme',
                             **{'min_value':-1.0, 'max_value':1.0})

        self.add('multiparam_tmean', multiparam_tmean_aggregation,
                 a={'min_value':0.0, 'max_value':0.5})
        self.add('max_median_min', max_median_min_aggregation,
                 a={'min_value':-1.0, 'max_value':1.0})
        self.add('maxabs_mean', maxabs_mean_aggregation,
                 a={'min_value':0.0, 'max_value':1.0})
        self.add('maxabs_tmean', maxabs_tmean_aggregation) # + extreme
        self.add('sum_mean', sum_mean_aggregation) # + average
        self.add('product_mean', product_mean_aggregation) # + average, use_median_sign
        self.add('sum_product', sum_product_aggregation) # + add_mult
        self.add('sum_product_mean', sum_product_mean_aggregation) # + use_median_sign, add_mult
        self.add('sum_maxabs_tmean', sum_maxabs_tmean_aggregation) # + average, extreme

    def add(self, name, function, **kwargs):
        self.multiparameterset.add_func(name, function, 'aggregation', **kwargs)

    def add_shared_name(self, name, **kwargs):
        self.multiparameterset.add_shared_name(name, 'aggregation', **kwargs)

    def get(self, name):
        return self.multiparameterset.get_func(name, 'aggregation')

    def __getitem__(self, index):
        warnings.warn(
            "Use get, not indexing ([{!s}]), for aggregation functions".format(saferepr(index)),
            DeprecationWarning)
        return self.get(index)

    def is_valid(self, name):
        return self.multiparameterset.is_valid_func(name, 'aggregation')

