"""Commonly used functions not available in the Python2 standard library."""
from __future__ import division

import math

from sys import float_info

NORM_EPSILON = math.pow(float_info.epsilon, 0.25) # half-precision works for machine learning

def mean(values):
    values = list(values)
    return math.fsum(map(float, values)) / len(values)

def median(values):
    values = list(values)
    values.sort()
    return values[len(values) // 2]

def median2(values):
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

def tmean(values, trim=0.25):
    """
    Returns the trimmed mean of the input values,
    with the fraction trimmed from each end being the second argument;
    requires 0.0 <= trim <= 0.5. If ``trim`` is over 0.25, returns the
    weighted mean of tmean(values, 0.25) and median2(values).
    """
    values = list(values)
    if (len(values) < 3) or (not trim):
        return mean(values)
    elif trim == 0.5:
        return median2(values)
    elif not (0.0 < trim < 0.5):
        raise ValueError(
            "Trim must be in 0.0 - 0.5 range, not {0!r}".format(trim))
    values.sort()
    if trim > 0.25: # trimming more than 50% of the values does not make much sense
        prop_trim = (0.5-trim)/0.25
        return (prop_trim*tmean(values, 0.25))+((1.0-prop_trim)*median2(values))
    orig_len = len(values)
    trim_fully = int(math.floor(trim*orig_len))
    trim_partially = trim*orig_len
    if trim_fully:
        values = values[trim_fully:]
        values = values[:-1*trim_fully]
    center_values = values
    if (len(center_values) < 3):
        return mean(center_values)
    if (trim_partially > trim_fully):
        center_values = center_values[1:]
        center_values = center_values[:-1]
    curr_sum = math.fsum(map(float,center_values))
    div_by = len(center_values)
    
    if (trim_partially > trim_fully):
        curr_sum += values[0]*(trim_partially-trim_fully)
        curr_sum += values[-1]*(trim_partially-trim_fully)
        div_by += 2*(trim_partially-trim_fully)
    return curr_sum/div_by

def variance(values):
    values = list(values)
    m = mean(values)
    return math.fsum((v - m) ** 2 for v in values) / len(values)


def stdev(values):
    return math.sqrt(variance(values))


def softmax(values):
    """
    Compute the softmax of the given value set, v_i = exp(v_i) / s,
    where s = sum(exp(v_0), exp(v_1), ..).
    """
    e_values = list(map(math.exp, values))
    s = sum(e_values)
    inv_s = 1.0 / s
    return [ev * inv_s for ev in e_values]


# Lookup table for commonly used {value} -> value functions.
stat_functions = {'min': min, 'max': max, 'mean': mean, 'median': median,
                  'median2': median2, 'tmean': tmean}
