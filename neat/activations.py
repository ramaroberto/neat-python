"""
Has the built-in activation functions,
methods for using them,
and methods for adding new user-defined ones
"""
from __future__ import division

import math
import warnings

from neat.math_util import NORM_EPSILON
from neat.multiparameter import MultiParameterSet
from neat.multiparameter import BadFunctionError as InvalidActivationFunction # pylint: disable=unused-import

def sigmoid_activation(z):
    z = max(-60.0, min(60.0, 5.0 * z))
    return 1.0 / (1.0 + math.exp(-z))


def tanh_activation(z):
    z = max(-60.0, min(60.0, 2.5 * z))
    return math.tanh(z)


def sin_activation(z):
    z = max(-60.0, min(60.0, 5.0 * z))
    return math.sin(z)


def gauss_activation(z):
    z = max(-3.4, min(3.4, z))
    return math.exp(-5.0 * z**2)


def relu_activation(z):
    return z if z > 0.0 else 0.0


def softplus_activation(z):
    z = max(-60.0, min(60.0, 5.0 * z))
    return 0.2 * math.log(1 + math.exp(z))


def identity_activation(z):
    return z


def clamped_activation(z):
    return max(-1.0, min(1.0, z))


def inv_activation(z):
    try:
        z = 1.0 / z
    except ArithmeticError: # handle overflows
        return 0.0
    else:
        return z


def log_activation(z):
    z = max(1e-7, z)
    return math.log(z)


def expanded_log_activation(z): # mostly intended for CPPNs
    if abs(z*2) < NORM_EPSILON:
        z = math.copysign((NORM_EPSILON/2),z)
    return math.copysign(1.0,z)*math.log(abs(z*2),2)


def skewed_log_plus_activation(z): # mostly intended for CPPNs
    return math.copysign(1.0,z)*(math.log1p(abs(z*2))-1)

def log_plus_activation(z):
    return math.copysign(1.0,z)*math.log1p(abs(z*math.sqrt(math.exp(1))))


def exp_activation(z):
    z = max(-60.0, min(60.0, z))
    return math.exp(z)


def abs_activation(z):
    return abs(z)


def hat_activation(z):
    return max(0.0, 1 - abs(z))


def square_activation(z):
    return z ** 2


def cube_activation(z):
    return z ** 3

def step_activation(z):
    if z < 0:
        return -1
    if z > 0:
        return 1
    return z


def multiparam_relu_activation(z, a):
    return max(z, (z*a))

def multiparam_elu_activation(z, a, b):
    z = min(60.0, max(-60.0, (5*z)))
    return 0.2 * min(abs(z), max(z, (math.exp(a)*(math.exp(z+b)-math.exp(b)))))

##def multiparam_lu_activation(z, a, b, c): # TEST NEEDED!
##    z = min(60.0, max(-60.0, (5*z)))
##    return 0.2 * min(abs(z), max(z, (z*a), (math.exp(b)*(math.exp(z+c)-math.exp(c)))))

##def multiparam_lu_partial_activation(z, a, b): # TEST NEEDED!
##    return multiparam_lu_activation(z, a, b, 0.0)

def weighted_lu_activation(z, a, b):
    assert a <= 1.0
    assert a >= 0.0

    return ((a*multiparam_relu_activation(z, b))+
            ((1-a)*multiparam_elu_activation(z, b, 0.0)))

def multiparam_relu_softplus_activation(z, a, b):
    assert a <= 1.0
    assert a >= 0.0
    assert b <= 1.0
    assert b >= 0.0

    val1 = ((a*relu_activation(z))+
            ((1-a)*z))
    val2 = ((a*abs(z))+
            ((1-a)*softplus_activation(z)))
    return ((b*val1)+((1-b)*val2))

def clamped_tanh_step_activation(z, a):
    assert a <= 1.0
    assert a >= -1.0
    tanh_weight = 1.0-abs(a)

    if a > 0.0:
        to_return = (((1.0-tanh_weight)*clamped_activation(z)) +
                     (tanh_weight*tanh_activation(z)))
    elif a < 0.0:
        to_return = (((1.0-tanh_weight)*step_activation(z)) +
                     (tanh_weight*tanh_activation(z)))
    else:
        to_return = tanh_activation(z)

    return max(-1.0,min(1.0,to_return))


def multiparam_sigmoid_activation(z, a):
    """Conversion of clamped_tanh_step_activation to a 0-1 output range"""
    return max(0.0,min(1.0,((clamped_tanh_step_activation(z, a)+1.0)/2.0)))


class ActivationFunctionSet(object):
    """Contains activation functions and methods to add and retrieve them."""
    def __init__(self, multiparameterset=None):
        if multiparameterset is None:
            warn_string = ("Activation init called without multiparameterset:" +
                           " may cause multiple instances of it")
            multiparameterset = MultiParameterSet('activation')
            warnings.warn(warn_string)
        self.multiparameterset = multiparameterset
        self.add('sigmoid', sigmoid_activation)
        self.add('tanh', tanh_activation)
        self.add('sin', sin_activation)
        self.add('gauss', gauss_activation)
        self.add('relu', relu_activation)
        self.add('softplus', softplus_activation)
        self.add('identity', identity_activation)
        self.add('clamped', clamped_activation)
        self.add('inv', inv_activation)
        self.add('log', log_activation)
        self.add('expanded_log', expanded_log_activation)
        self.add('log_plus', log_plus_activation)
        self.add('skewed_log_plus', skewed_log_plus_activation)
        self.add('exp', exp_activation)
        self.add('abs', abs_activation)
        self.add('hat', hat_activation)
        self.add('square', square_activation)
        self.add('cube', cube_activation)
        self.add('step', step_activation)
        self.add('multiparam_relu', multiparam_relu_activation,
                 a={'min_value':-1.0, 'max_value':1.0})
        self.add('multiparam_elu', multiparam_elu_activation,
                 a={'min_value':-1.0, 'max_value':1.0},
                 b={'min_value':-1.0, 'max_value':1.0})
##        self.add('multiparam_lu', multiparam_lu_activation,
##                 a={'min_value':-1.0, 'max_value':1.0},
##                 b={'min_value':-1.0, 'max_value':1.0},
##                 c={'min_value':-1.0, 'max_value':1.0})
##        self.add('multiparam_lu_partial', multiparam_lu_partial_activation,
##                 a={'min_value':-1.0, 'max_value':1.0},
##                 b={'min_value':-1.0, 'max_value':1.0})
        self.add('weighted_lu', weighted_lu_activation,
                 a={'min_value':0.0, 'max_value':1.0},
                 b={'min_value':-1.0, 'max_value':1.0})
        self.add('multiparam_relu_softplus', multiparam_relu_softplus_activation,
                 a={'min_value':0.0, 'max_value':1.0},
                 b={'min_value':0.0, 'max_value':1.0})
        self.add('clamped_tanh_step', clamped_tanh_step_activation,
                 a={'min_value':-1.0, 'max_value':1.0})
        self.add('multiparam_sigmoid', multiparam_sigmoid_activation,
                 a={'min_value':-1.0, 'max_value':1.0})
        

    def add(self, name, function, **kwargs):
        self.multiparameterset.add_func(name, function, 'activation', **kwargs)

    def get(self, name):
        return self.multiparameterset.get_func(name, 'activation')

    def __getitem__(self, index):
        warnings.warn("Use get, not indexing ([{!r}]), for activation functions".format(index),
                      DeprecationWarning)
        return self.get(index)

    def is_valid(self, name):
        return self.multiparameterset.is_valid_func(name, 'activation')
