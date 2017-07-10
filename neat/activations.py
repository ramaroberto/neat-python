"""
Has the built-in activation functions, methods for using them,
and methods for adding new user-defined ones.
"""
from __future__ import division

import math
import warnings

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

def multiparam_relu_activation(z, a):
    return max(z, (z*a))

class ActivationFunctionSet(object):
    """Contains activation functions and methods to add and retrieve them."""
    def __init__(self, multiparameterset):
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
        self.add('multiparam_relu', multiparam_relu_activation, a={'min_value':-1, 'max_value':1})

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
