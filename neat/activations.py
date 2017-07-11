"""
Has the built-in activation functions, methods for using them,
and methods for adding new user-defined ones.
"""
from __future__ import division

import math
import warnings
from neat.multiparameter import MultiParameterSet

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
    assert a <= 1.0
    assert a >= -1.0
    return max(z, (z*a))


def clamped_tanh_step_activation(z, a):
    assert a <= 1.0
    assert a >= -1.0
    tanh_weight = 1.0-abs(a)

    if a > 0.0:
        return (((1.0-tanh_weight)*clamped_activation(z)) +
                (tanh_weight*tanh_activation(z)))
    if a < 0.0:
        return (((1.0-tanh_weight)*step_activation(z)) +
                (tanh_weight*tanh_activation(z)))
    
    return tanh_activation(z)


def multiparam_sigmoid_activation(z, a):
    """Conversion of clamped_tanh_step_activation to a 0-1 output range"""
    return max(0.0,min(1.0,((clamped_tanh_step_activation(z, a)+1.0)/2.0)))


class ActivationFunctionSet(object):
    """Contains activation functions and methods to add and retrieve them."""
    def __init__(self, multiparameterset=None):
        if multiparameterset is None:
            warnings.warn("Activation init called without multiparameterset:"+
                          " may cause multiple instances of it")
            multiparameterset = MultiParameterSet('activation')
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
        self.add('exp', exp_activation)
        self.add('abs', abs_activation)
        self.add('hat', hat_activation)
        self.add('square', square_activation)
        self.add('cube', cube_activation)
        self.add('step', step_activation)
        self.add('multiparam_relu', multiparam_relu_activation,
                 a={'min_value':-1.0, 'max_value':1.0})
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
