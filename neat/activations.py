"""
Has the built-in activation functions,
methods for using them,
and methods for adding new user-defined ones
"""
from __future__ import division

import math
import sys
import warnings

from pprint import saferepr

from neat.math_util import NORM_EPSILON
from neat.multiparameter import MultiParameterSet
from neat.multiparameter import BadFunctionError as InvalidActivationFunction # pylint: disable=unused-import

def sigmoid_activation(z):
    try:
        to_return = 1.0 / (1.0 + math.exp(-4.9*z))
    except ArithmeticError: # pragma: no cover
        if z > 0.0:
            return 1.0
        return 0.0
    else:
        return to_return


def tanh_activation(z):
    try:
        to_return = math.tanh(z*2.45)
    except ArithmeticError: # pragma: no cover
        if z > 0.0:
            return 1.0
        return -1.0
    else:
        return to_return


def sin_activation(z):
    z = max(-60.0, min(60.0, 5.0 * z))
    return math.sin(z)


def gauss_activation(z):
    try:
        to_return = math.exp(-5.0 * z**2)
    except ArithmeticError: # pragma: no cover
        if abs(z) > NORM_EPSILON:
            return 0.0
        return 1.0
    else:
        return to_return


def relu_activation(z):
    return z if z > 0.0 else 0.0


def softplus_activation(z):
    try:
        to_return = 0.2 * math.log1p(math.exp(z*5.0))
    except ArithmeticError: # pragma: no cover
        if z > NORM_EPSILON:
            return z
        elif z < -1*NORM_EPSILON:
            return 0.0
        return (NORM_EPSILON/2.0)
    else:
        return to_return


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
    z = max(sys.float_info.epsilon, z)
    try:
        to_return = math.log(z)
    except ArithmeticError: # pragma: no cover
        return math.log(1e-7)
    else:
        return to_return


def expanded_log_activation(z): # mostly intended for CPPNs
    if abs(z*2) < NORM_EPSILON:
        z = math.copysign((NORM_EPSILON/2),z)
    return math.copysign(1.0,z)*math.log(abs(z*2),2)


def skewed_log1p_activation(z): # mostly intended for CPPNs
    return math.copysign(1.0,z)*(math.log1p(abs(z*2))-1)

def log1p_activation(z):
    return math.copysign(1.0,z)*math.log1p(abs(z*math.exp(0.5)))


def exp_activation(z):
    z = max(-60.0, min(60.0, z))
    return math.exp(z)


def abs_activation(z):
    return abs(z)


def hat_activation(z):
    return max(0.0, 1.0 - abs(z))


def square_activation(z):
    return z ** 2


def cube_activation(z):
    return z ** 3

def step_activation(z):
    if z < 0.0:
        return -1
    if z > 0.0:
        return 1
    return z

def square_wave_activation(z):
    return step_activation(sin_activation(z))

def triangle_wave_activation(z):
    return min(1.0,max(-1.0,((2/math.pi)*math.asin(sin_activation(z)))))

def rectangular_activation(z):
    if abs(z) == 0.5:
        return 0.5
    if abs(z) < 0.5:
        return 1.0
    return 0.0

def multiparam_relu_activation(z, a):
    return max(z, (z*a))

def multiparam_elu_activation_inner(z, a, b):
    try:
        result = 0.2 * min(abs(z*5), max((z*5), (math.exp(a)*(math.exp((z*5)+b)-math.exp(b)))))
    except ArithmeticError:
        old_z = z
        old_a = a
        old_b = b
        a = min(sys.float_info.max_10_exp, max(sys.float_info.min_10_exp, a))
        b = min(sys.float_info.max_10_exp, max(sys.float_info.min_10_exp, a))
        z = min(((60.0-b)/math.exp(a)), max(((-60.0-b)/math.exp(a)), z))
        if (abs(z) < abs(old_z)) or (abs(a) < abs(old_a)) or (abs(b) < abs(old_b)):
            return multiparam_elu_activation_inner(z, a, b)
        return multiparam_elu_activation_inner(old_z, min(2.0,max(-1.0,a)),
                                               min(2.0,max(-2.0,b)))
    else:
        return result

def multiparam_elu_activation(z, a, b):
    a_use = min(sys.float_info.max_10_exp, max(sys.float_info.min_10_exp, a))
    return multiparam_elu_activation_inner((z*min(1.0,math.exp(-a_use))), a, b)

##def multiparam_elu_variant_activation(z, a, b):
##    try:
##        result = 0.2 * min(abs(z*5), max((z*5), (math.exp((z+a)*5)-math.exp(b))))
##    except ArithmeticError:
##        z = 0.2*min(300.0, max(-300.0, (z*5)))
##        return multiparam_elu_variant_activation(z, a, b)
##    else:
##        return result

##def multiparam_lu_activation(z, a, b, c): # TEST NEEDED!
##    z = min(60.0, max(-60.0, (5*z)))
##    return 0.2 * min(abs(z), max(z, (z*a), (math.exp(b)*(math.exp(z+c)-math.exp(c)))))

##def multiparam_lu_partial_activation(z, a, b): # TEST NEEDED!
##    return multiparam_lu_activation(z, a, b, 0.0)

def _check_value_range(a, min_val, max_val, caller, var_name):
    if not min_val <= a <= max_val:
        raise ValueError(
            "{0} for {1}_activation must be between {2:n} and {3:n}, not {4!s}".format(
                var_name, caller, min_val, max_val, saferepr(a)))

def weighted_lu_activation(z, a, b):
    _check_value_range(a, 0.0, 1.0, 'weighted_lu', 'a')

    return ((a*multiparam_relu_activation(z, b))+
            ((1-a)*multiparam_elu_activation_inner(z, b, 0.0)))

def multiparam_relu_softplus_activation(z, a, b):
    _check_value_range(a, 0.0, 1.0, 'multiparam_relu_softplus', 'a')
    _check_value_range(b, 0.0, 1.0, 'multiparam_relu_softplus', 'b')

    val1 = ((a*relu_activation(z))+
            ((1.0-a)*z))
    val2 = ((a*abs(z))+
            ((1.0-a)*softplus_activation(z)))
    return ((b*val1)+((1.0-b)*val2))

def clamped_tanh_step_activation(z, a):
    _check_value_range(a, -1.0, 1.0, 'clamped_tanh_step', 'a')
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

ABS_GAUSS_EPSILON = math.exp(-5.0 * NORM_EPSILON)

def multiparam_gauss_activation(z, a, b):
    _check_value_range(a, 0.0, 4.0, 'multiparam_gauss', 'a')
    _check_value_range(b, 0.0, 2.0, 'multiparam_gauss', 'b')

    mult = 1.0
    if a < 1.0:
        a = math.exp(a-1.0)
        test_value = math.exp(-5.0 * NORM_EPSILON**a)
        if test_value < ABS_GAUSS_EPSILON:
            mult = ABS_GAUSS_EPSILON/test_value

    if b < 0.5:
        b = 0.5*math.exp(b-0.5)

    try:
        to_return = min(1.0,(mult*math.exp(-5.0 * b * abs(z)**a)))
    except ArithmeticError: # pragma: no cover
        if abs(z) > NORM_EPSILON:
            return 0.0
        return 1.0
    else:
        return to_return

def hat_gauss_rectangular_activation(z, a, b):
    _check_value_range(a, 0.0, 1.0, 'hat_gauss_rectangular', 'a')
    _check_value_range(b, 0.0, 1.0, 'hat_gauss_rectangular', 'b')

    rectangular_weight = (1.0-b)*a
    hat_weight = (1.0-b)*(1.0-a)

    to_return = 0.0

    if hat_weight > 0.0:
        to_return += hat_weight*hat_activation(z)
    if b > 0.0:
        to_return += b*multiparam_gauss_activation(z, (a*4.0), 1.0)
    if rectangular_weight > 0.0:
        to_return += rectangular_weight*rectangular_activation(z)

    return max(-1.0,min(1.0,to_return))

def scaled_expanded_log_activation(z, a): # mostly intended for CPPNs
    a = min(sys.float_info.max_10_exp, max(sys.float_info.min_10_exp, a))
    if abs(z*math.pow(2.0,abs(a))) < NORM_EPSILON:
        z = math.copysign((NORM_EPSILON/math.pow(2.0,abs(a))),z)
    return math.copysign(math.pow(2.0,(1.0-a)),z)*math.log(abs(z*math.pow(2.0,abs(a))),2)

def multiparam_log_inv_activation(z, a): # mostly intended for CPPNs
    assert a >= -1.0, "'a' for multiparam_log_inv must be -1.0 or above, not {!s}".format(
        saferepr(a))
    if a >= 0:
        return scaled_expanded_log_activation(z,(a+1.0))
    return ((abs(a)*inv_activation(-1*z))
            +((1.0-abs(a))*scaled_expanded_log_activation(z,1.0)))

def scaled_log1p_activation(z, a):
    a = min(sys.float_info.max_10_exp, max(sys.float_info.min_10_exp, a))
    return math.copysign(math.exp(0.5-a),z)*math.log1p(abs(z*math.exp(a)))

def multiparam_tanh_log1p_activation(z, a, b):
    _check_value_range(a, 0.0, 1.0, 'multiparam_tanh_log1p', 'a')
    _check_value_range(b, -1.0, 1.0, 'multiparam_tanh_log1p', 'b')

    tanh_part = a*clamped_tanh_step_activation(z, b)

    if b > 0.0:
        other_part = (1.0-a)*((b*z)+
                              ((1.0-b)*scaled_log1p_activation(z, 0.5)))
    else:
        other_part = (1.0-a)*scaled_log1p_activation(z, (0.5-(1.5*b)))

    return tanh_part+other_part

def multiparam_pow_activation(z, a):
    if a < 1.0:
        a = math.pow(2,(a-1.0))
    return math.copysign(1.0,z)*math.pow(abs(z),a)

##def multiparam_exp_activation(z, a):
##    _check_value_range(a, 0.0, 1.0, 'multiparam_exp', 'a')

##    a = min(1.0,((a*(1-NORM_EPSILON)) + NORM_EPSILON))

##    if z >= math.log(a):
##        return math.exp(z)
##    return -1.0*math.exp(math.log(a)-z)

def wave_activation(z, a):
    _check_value_range(a, -1.0, 1.0, 'wave', 'a')
    sin_wgt = 1.0-abs(a)

    if a > 0.0:
        to_return = ((a*triangle_wave_activation(z))
                     +(sin_wgt*sin_activation(z)))
    elif a < 0.0:
        to_return = (((1-sin_wgt)*square_wave_activation(z))
                     +(sin_wgt*sin_activation(z)))
    else:
        to_return = sin_activation(z)

    return min(1.0,max(-1.0,to_return))
        
def multiparam_tanh_approx_activation(z, a, b):
    _check_value_range(b, 0.0, 1.0, 'multiparam_tanh_approx', 'b')

    try:
        to_return = (2.5*z)/(math.exp(a) + (b*abs(2.45*z)))
    except ArithmeticError: # pragma: no cover
        _check_value_range(a, -12.0, 12.0, 'multiparam_tanh_approx', 'a')
        raise
    else:
        return to_return

def multiparam_sigmoid_approx_activation(z, a):
    return min(1.0,max(0.0,((1.0+multiparam_tanh_approx_activation((2.0*z), a, 1.0))/2.0)))

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
        self.add('log1p', log1p_activation)
        self.add('skewed_log1p', skewed_log1p_activation)
        self.add('exp', exp_activation)
        self.add('abs', abs_activation)
        self.add('hat', hat_activation)
        self.add('square', square_activation)
        self.add('cube', cube_activation)
        self.add('step', step_activation)
        self.add('square_wave', square_wave_activation)
        self.add('triangle_wave', triangle_wave_activation)
        self.add('rectangular', rectangular_activation)
        self.add('multiparam_relu', multiparam_relu_activation,
                 a={'min_value':-1.0, 'max_value':1.0})
##        self.add('multiparam_elu', multiparam_elu_activation_inner,
##                 a={'min_value':-1.0, 'max_value':1.0},
##                 b={'min_value':-1.0, 'max_value':1.0})
        self.add('multiparam_elu', multiparam_elu_activation,
                 a={'min_value':0.0, 'max_value':1.0},
                 b={'min_value':-1.0, 'max_value':1.0})
##        self.add('multiparam_elu_variant', multiparam_elu_variant_activation,
##                 a={'min_value':-1.0, 'max_value':1.0},
##                 b={'min_value':-1.0, 'max_value':1.0})
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
        self.add('hat_gauss_rectangular', hat_gauss_rectangular_activation,
                 a={'min_value':0.0, 'max_value':1.0},
                 b={'min_value':0.0, 'max_value':1.0})
        self.add('scaled_expanded_log', scaled_expanded_log_activation,
                 a={'min_value':0.0, 'max_value':2.0})
        self.add('multiparam_log_inv', multiparam_log_inv_activation,
                 a={'min_value':-1.0, 'max_value':1.0})
        self.add('scaled_log1p', scaled_log1p_activation,
                 a={'min_value':0.0, 'max_value':2.0})
        self.add('multiparam_tanh_log1p', multiparam_tanh_log1p_activation,
                 a={'min_value':0.0, 'max_value':1.0},
                 b={'min_value':-1.0, 'max_value':1.0})
        self.add('multiparam_pow', multiparam_pow_activation,
                 a={'min_init_value':-1.0, 'max_init_value':4.0,
                    'min_value':-3.0, 'max_value':16.0,
                    'init_type':'gaussian'})
##        self.add('multiparam_exp', multiparam_exp_activation,
##                 a={'min_value':0.0, 'max_value':1.0})
        self.add('wave', wave_activation,
                 a={'min_value':-1.0, 'max_value': 1.0})
        self.add('multiparam_tanh_approx', multiparam_tanh_approx_activation,
                 a={'min_init_value':-1.0, 'max_init_value':1.0,
                    'min_value':-12.0, 'max_value':12.0,
                    'init_type':'gaussian'},
                 b={'min_value':0.0, 'max_value':1.0})
        self.add('multiparam_sigmoid_approx', multiparam_sigmoid_approx_activation,
                 a={'min_init_value':-1.0, 'max_init_value':1.0,
                    'min_value':-12.0, 'max_value':12.0,
                    'init_type':'gaussian'})
        self.add('multiparam_gauss', multiparam_gauss_activation,
                 a={'min_value':0.0, 'max_value':4.0},
                 b={'min_value':0.0, 'max_value':2.0})


    def add(self, name, function, **kwargs):
        self.multiparameterset.add_func(name, function, 'activation', **kwargs)

    def get(self, name):
        return self.multiparameterset.get_func(name, 'activation')

    def __getitem__(self, index):
        warnings.warn(
            "Use get, not indexing ([{!s}]), for activation functions".format(saferepr(index)),
            DeprecationWarning)
        return self.get(index)

    def is_valid(self, name):
        return self.multiparameterset.is_valid_func(name, 'activation')
