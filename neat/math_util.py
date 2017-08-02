"""Commonly used functions not available in the Python2 standard library."""
from __future__ import division

from math import sqrt, exp

from neat.mypy_util import * # pylint: disable=unused-wildcard-import

if MYPY: # pragma: no cover
    from typing import Callable # pylint: disable=unused-import

def mean(values): # type: (Iterable[float]) -> float
    values = list(values)
    return sum(map(float, values)) / len(values)


def median(values): # type: (Iterable[float]) -> float
    values = list(values)
    values.sort()
    return values[len(values) // 2]


def median2(values): # type: (Iterable[float]) -> float
    """
    Returns the median of the input values;
    if there are an even number of inputs, returns the mean of the middle two.
    """
    values = list(values)
    n = len(values)
    if n <= 2:
        return mean(values)
    values.sort()
    if (n % 2) == 1:
        return values[n//2]
    i = n//2
    return (values[i - 1] + values[i])/2.0

def variance(values): # type: (Iterable[float]) -> float
    values = list(values)
    m = mean(values)
    return sum((v - m) ** 2 for v in values) / len(values)


def stdev(values): # type: (Iterable[float]) -> float
    return sqrt(variance(values))


def softmax(values): # type: (Iterable[float]) -> Iterable[float]
    """
    Compute the softmax of the given value set, v_i = exp(v_i) / s,
    where s = sum(exp(v_0), exp(v_1), ..).
    """
    e_values = list(map(exp, values))
    s = sum(e_values)
    inv_s = 1.0 / s
    return [ev * inv_s for ev in e_values]


# Lookup table for commonly used {value} -> value functions.
stat_functions = {'min': min, 'max': max, 'mean': mean, 'median': median,
                  'median2': median2} # type: Dict[str, Callable[[Iterable[float]], float]]
