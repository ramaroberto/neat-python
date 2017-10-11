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

from neat.math_util import NORM_EPSILON, check_value_range
from neat.multiparameter import MultiParameterSet
from neat.multiparameter import BadFunctionError as InvalidActivationFunction # pylint: disable=unused-import

def sigmoid_activation(z):
    try:
        to_return = 1.0 / (1.0 + math.exp(-5*z))
    except ArithmeticError: # pragma: no cover
        if z > 0.0:
            return 1.0
        return 0.0
    else:
        return to_return


def tanh_activation(z):
    try:
        to_return = math.tanh(z*2.5)
    except ArithmeticError: # pragma: no cover
        if z > 0.0:
            return 1.0
        return -1.0
    else:
        return to_return

_TANH_APPROX_ADD = math.pow(5.887823,-0.239897)

def tanh_approx_activation(z):
    if z >= (sys.float_info.max/5.887823):
        return 1.0
    if z <= (-1*sys.float_info.max/5.887823):
        return -1.0
    to_return = (5.887823*z)/(_TANH_APPROX_ADD + abs(5.887823*z))
    return min(1.0,max(-1.0,(1.0222006359744715*to_return)))

def sigmoid_approx_activation(z):
    return min(1.0,max(0.0,((1.0+tanh_approx_activation(1.1054759*z))/2.0)))

def sin_activation(z):
    z = max(-60.0, min(60.0, 5.0 * z))
    return math.sin(z)

def gauss_activation(z):
    try:
        to_return = math.exp(-5.0 * z**2)
    except ArithmeticError: # pragma: no cover
        if abs(z) > 0.5:
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
        to_return = 1.0 / z
    except ArithmeticError: # handle overflows
        return 0.0
    else:
        return to_return


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

def multiparam_relu_activation(z, tilt):
    return max(z, (z*tilt))

def _multiparam_elu_activation_inner(z, a, b):
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
            return _multiparam_elu_activation_inner(z, a, b)
        return _multiparam_elu_activation_inner(old_z, min(2.0,max(-1.0,a)),
                                                min(2.0,max(-2.0,b)))
    else:
        return result

def multiparam_elu_activation(z, tilt, lower):
    a = (1.0-tilt)/2.0
    b = (2.0*lower)-1.0
    a_use = min(sys.float_info.max_10_exp, max(sys.float_info.min_10_exp, a))
    return _multiparam_elu_activation_inner((z*min(1.0,math.exp(-a_use))), a, b)

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
    check_value_range(var=a,
                      min_val=min_val,
                      max_val=max_val,
                      caller=caller,
                      var_name=var_name,
                      add_name="activation")

def weighted_lu_activation(z, curve, tilt):
    _check_value_range(curve, 0.0, 1.0, 'weighted_lu', 'curve')

    if curve == 0.0:
        return multiparam_relu_activation(z, tilt)
    if curve == 1.0:
        return _multiparam_elu_activation_inner(z, tilt, 0.0)

    return (((1.0-curve)*multiparam_relu_activation(z, tilt))+
            (curve*_multiparam_elu_activation_inner(z, tilt, 0.0)))

def _multiparam_softplus_activation_inner(z, curve):
    a = 1.0 - (2.5*curve)
    try:
        to_return = math.log1p(math.exp(5.0*z*math.exp(a)))*0.2*math.exp(-a)
    except ArithmeticError: # pragma: no cover
        if z >= NORM_EPSILON:
            return z
        elif z <= -1*NORM_EPSILON:
            return 0.0
        return (z+NORM_EPSILON)/2.0
    else:
        return to_return

def multiparam_softplus_activation(z, curve):
    if curve >= 0.5:
        return _multiparam_softplus_activation_inner(z, curve)
    elif curve == 0.0:
        return relu_activation(z)

    weight_relu = ((0.5-curve)*2.0)**4

    return (((1.0-weight_relu)*_multiparam_softplus_activation_inner(z, curve))+
            (weight_relu*relu_activation(z)))

def multiparam_relu_softplus_activation(z, tilt, curve):
    _check_value_range(tilt, -1.0, 1.0, 'multiparam_relu_softplus', 'tilt')
    _check_value_range(curve, 0.0, 1.0, 'multiparam_relu_softplus', 'curve')

    weight_softplus = (1.0-abs(tilt)+curve)/2.0
    if weight_softplus < NORM_EPSILON:
        return multiparam_relu_activation(z, tilt)
    if weight_softplus == 1.0:
        return _multiparam_softplus_activation_inner(z, curve)

##    lower_sub = abs(tilt)*1.0
##    lower_use = (lower*(1.0+lower_sub)) - lower_sub

    return ((weight_softplus*_multiparam_softplus_activation_inner(z, curve))
            + ((1.0-weight_softplus)*multiparam_relu_activation(z, tilt)))

##    a = (1.0-tilt)/2.0

##    val1 = ((a*relu_activation(z))+
##            ((1.0-a)*z))
##    val2 = ((a*abs(z))+
##            ((1.0-a)*softplus_activation(z)))
##    return ((lower*min(val1,val2))+((1.0-lower)*max(val1,val2)))

def clamped_tanh_step_activation(z, tilt):
    _check_value_range(tilt, -1.0, 1.0, 'clamped_tanh_step', 'tilt')
    tanh_weight = 1.0-abs(tilt)

    if tilt > 0.0:
        if tilt == 1.0:
            return clamped_activation(z)
        to_return = (((1.0-tanh_weight)*clamped_activation(z)) +
                     (tanh_weight*tanh_activation(z)))
    elif tilt < 0.0:
        if tilt == -1.0:
            return step_activation(z)
        to_return = (((1.0-tanh_weight)*step_activation(z)) +
                     (tanh_weight*tanh_activation(z)))
    else:
        to_return = tanh_activation(z)

    return max(-1.0,min(1.0,to_return))

def multiparam_sigmoid_activation(z, tilt):
    """Conversion of clamped_tanh_step_activation to a 0-1 output range"""
    return max(0.0,min(1.0,((clamped_tanh_step_activation(z, tilt)+1.0)/2.0)))

def clamped_step_activation(z, tilt):
    _check_value_range(tilt, -1.0, 1.0, 'clamped_tanh_step', 'tilt')

    if tilt == 1.0:
        return clamped_activation(z)
    if tilt == -1.0:
        return step_activation(z)
    if z == 0.0:
        return 0.0

    a=(1.0+tilt)/2.0
    if abs(z) >= a:
        return step_activation(z)
    return z/a

def _extended_clamped_step_activation(z, tilt2):
    if tilt2 <= -1.0:
        return step_activation(z)
    if z == 0.0:
        return 0.0

    a=(1.0+tilt2)/2.0
    if abs(z) >= a:
        return step_activation(z)
    return z/a

_ABS_GAUSS_EPSILON = math.exp(-5.0 * NORM_EPSILON)

def multiparam_gauss_activation(z, width, lower):
    _check_value_range(width, 0.0, 1.0, 'multiparam_gauss', 'width')
    _check_value_range(lower, 0.0, 1.0, 'multiparam_gauss', 'lower')

    a = width*4.0
    b = lower*2.0

    mult = 1.0
    if a < 1.0:
        a = math.exp(a-1.0)
        test_value = math.exp(-5.0 * NORM_EPSILON**a)
        if test_value < _ABS_GAUSS_EPSILON:
            mult = _ABS_GAUSS_EPSILON/test_value

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

def hat_gauss_rectangular_activation(z, width, curve):
    _check_value_range(width, 0.0, 1.0, 'hat_gauss_rectangular', 'width')
    _check_value_range(curve, 0.0, 1.0, 'hat_gauss_rectangular', 'curve')

    rectangular_weight = (1.0-curve)*width
    hat_weight = (1.0-curve)*(1.0-width)

    to_return = 0.0

    if hat_weight > 0.0:
        if hat_weight == 1.0:
            return hat_activation(z)
        to_return += hat_weight*hat_activation(z)
    if rectangular_weight > 0.0:
        if rectangular_weight == 1.0:
            return rectangular_activation(z)
        to_return += rectangular_weight*rectangular_activation(z)
    if curve > 0.0:
        to_return += curve*multiparam_gauss_activation(z, width, 0.5)

    return max(-1.0,min(1.0,to_return))

def scaled_expanded_log_activation(z, tilt): # mostly intended for CPPNs
    a = 1.0-tilt
    a = min(sys.float_info.max_10_exp, max(sys.float_info.min_10_exp, a))
    if abs(z*math.pow(2.0,abs(a))) < NORM_EPSILON:
        z = math.copysign((NORM_EPSILON/math.pow(2.0,abs(a))),z)
    return math.copysign(math.pow(2.0,(1.0-a)),z)*math.log(abs(z*math.pow(2.0,abs(a))),2)

def multiparam_log_inv_activation(z, tilt): # mostly intended for CPPNs
    #_check_value_range(tilt, -1.0, 1.0, 'multiparam_log_inv', 'tilt')
    if tilt >= 0.0:
        return scaled_expanded_log_activation(z,tilt)
    if tilt == -1.0:
        return inv_activation(-1*z)
    return ((abs(tilt)*inv_activation(-1*z))
            +((1.0-abs(tilt))*scaled_expanded_log_activation(z,0.0)))

def scaled_log1p_activation(z, tilt):
    a = (0.5-(1.5*tilt))
    a = min(sys.float_info.max_10_exp, max(sys.float_info.min_10_exp, a))
    return math.copysign(math.exp(0.5-a),z)*math.log1p(abs(z*math.exp(a)))

def clamped_log1p_step_activation(z, curve, tilt):
    #_check_value_range(curve, 0.0, 1.0, 'clamped_log1p_step', 'curve')
    #_check_value_range(tilt, -1.0, 1.0, 'clamped_log1p_step', 'tilt')

    tilt2 = ((tilt+1.0)*1.5)-1.0

    clamped_step_part = (1.0-curve)*_extended_clamped_step_activation(z, tilt2)
    if curve == 0.0:
        return clamped_step_part

    other_part = curve*scaled_log1p_activation(z, tilt)
    return clamped_step_part+other_part

##    if tilt >= 0.32220367278798001:
##        clamped_step_slope = (0.023736915154962276*tilt) + 0.76726263248551119
##    elif tilt <= -0.27545909849749584:
##        clamped_step_slope = (0.15249138843483784*tilt) + 0.75685312690039497
##    else:
##        prop_high = (tilt + 0.27545909849749584)/(0.32220367278798001 + 0.27545909849749584)
##        high_slope = (0.023736915154962276*tilt) + 0.76726263248551119
##        low_slope = (0.15249138843483784*tilt) + 0.75685312690039497
##        clamped_step_slope = (prop_high*high_slope)+((1.0-prop_high)*low_slope)
##    tilt_for_scaled_log1p = (clamped_step_slope - 0.85883776552031577)/0.49708835912756039
##    other_part = scaled_log1p_activation(z, tilt_for_scaled_log1p)
##    if curve == 1.0:
##        return other_part
##    return clamped_step_part+(curve*other_part)

def multiparam_tanh_log1p_activation(z, curve, tilt):
    #_check_value_range(curve, 0.0, 1.0, 'multiparam_tanh_log1p', 'curve')
    #_check_value_range(tilt, -1.0, 1.0, 'multiparam_tanh_log1p', 'tilt')

    tanh_part = (1.0-curve)*clamped_tanh_step_activation(z, tilt)
    if curve == 0.0:
        return tanh_part

    if tilt > 0.0:
        other_part = curve*((tilt*z)+
                            ((1.0-tilt)*scaled_log1p_activation(z, 0.0)))
    else:
        other_part = curve*scaled_log1p_activation(z, tilt)

    return tanh_part+other_part

def multiparam_pow_activation(z, tilt):
    if tilt >= 0.0:
        a = (tilt*3.0)+1.0
    else:
        a = (tilt*2.0)+1.0
    if a < 1.0:
        a = math.pow(2,(a-1.0))
    return math.copysign(1.0,z)*math.pow(abs(z),a)

##def multiparam_exp_activation(z, a):
##    _check_value_range(a, 0.0, 1.0, 'multiparam_exp', 'a')

##    a = min(1.0,((a*(1-NORM_EPSILON)) + NORM_EPSILON))

##    if z >= math.log(a):
##        return math.exp(z)
##    return -1.0*math.exp(math.log(a)-z)

def wave_activation(z, width):
    #_check_value_range(width, 0.0, 1.0, 'wave', 'width')
    a = 1.0-(2*width)
    sin_wgt = 1.0-abs(a)

    if a > 0.0:
        if a == 1.0:
            return triangle_wave_activation(z)
        to_return = ((a*triangle_wave_activation(z))
                     +(sin_wgt*sin_activation(z)))
    elif a < 0.0:
        if a == -1.0:
            return square_wave_activation(z)
        to_return = (((1-sin_wgt)*square_wave_activation(z))
                     +(sin_wgt*sin_activation(z)))
    else:
        to_return = sin_activation(z)

    return min(1.0,max(-1.0,to_return))

def multiparam_tanh_approx_activation(z, g_curve, tilt):
    #_check_value_range(tilt, -1.0, 1.0, 'multiparam_tanh_approx', 'tilt')

    b = 1.0-tilt
    if b > 1.0:
        b **= 4

    try:
        to_return = (max(1.0,b)*5.887823*z)/(math.pow(5.887823,g_curve)
                                             + (b*abs(5.887823*z)))
    except ArithmeticError: # pragma: no cover
        if tilt < NORM_EPSILON:
            if z > NORM_EPSILON:
                return 1.0
            if z < -NORM_EPSILON:
                return -1.0
            return z
        _check_value_range(g_curve, -12.0, 12.0, 'multiparam_tanh_approx', 'g_curve')
        raise
    else:
        minmax = 1.0
        if tilt >= 1.0:
            return to_return*1.0222006359744715
        if tilt > sys.float_info.epsilon:
            minmax = inv_activation(1.0-tilt)
            if (minmax <= 0.0) or (minmax >= (10**sys.float_info.dig)): # pragma: no cover
                return to_return*1.0222006359744715
        return min(minmax,max(-minmax,(to_return*1.0222006359744715)))

def multiparam_sigmoid_approx_activation(z, g_curve):
    return min(1.0,max(0.0,((1.0+multiparam_tanh_approx_activation(
        (1.1054759*z), g_curve, 0.0))/2.0)))

##def bisigmoid_activation(z, tilt, width):
##    a = 5.0*tilt
##    part1 = math.exp(a)*(z + width)
##    part2 = math.exp(-a)*(z - width)

##    return sigmoid_approx_activation(part1)*(1.0-sigmoid_approx_activation(part2))

##def bicentral_activation(z, lower, tilt):
##    a = ((lower*-2.0)+1.0)*3.0
##    b = tilt*3.0
##    a = min((sys.float_info.max_10_exp/2.0), max((sys.float_info.min_10_exp/2.0), a))
##    b = min((sys.float_info.max_10_exp/2.0), max((sys.float_info.min_10_exp/2.0), b))
##    part1 = math.exp(a+b)*(z + 0.5)
##    part2 = math.exp(a-b)*(z - 0.5)

##    return sigmoid_approx_activation(part1)*(1.0-sigmoid_approx_activation(part2))

def bicentral_activation(z, lower, tilt, width):
    a = ((lower*-2.0)+1.0)*3.0
    b = tilt*3.0
    a = min((sys.float_info.max_10_exp/2.0), max((sys.float_info.min_10_exp/2.0), a))
    b = min((sys.float_info.max_10_exp/2.0), max((sys.float_info.min_10_exp/2.0), b))
    part1 = math.exp(a+b)*(z + width)
    part2 = math.exp(a-b)*(z - width)

    height_mult = min(1.0,max(abs(tilt),(1.5-lower)))

    return height_mult*sigmoid_approx_activation(part1)*(1.0-sigmoid_approx_activation(part2))

def fourth_square_abs_activation(z, width):
    #_check_value_range(width, 0.0, 1.0, 'fourth_square_abs', 'width')
    a = (width*2.0)-1.0

    weight_square = 1.0-abs(a)

    if a > 0.0:
        if a >= 1.0:
            return z**4
        return ((1.0-weight_square)*(z**4))+(weight_square*(z**2))
    if a < 0.0:
        if a <= -1.0:
            return abs(z)
        return ((1.0-weight_square)*abs(z))+(weight_square*(z**2))
    return z**2

def rational_quadratic_activation(z, lower, width):
    a = (lower*2.0) - 1.0
    b = width - 1.0
    alpha = math.exp(3.0*a)
    length_scale = math.exp(6.0*b)

    try:
        to_return = math.pow((1.0+
                              (math.pow((math.sqrt(2.0)*z),2.0)/
                               (2.0*alpha*math.pow(length_scale,2.0)))),
                             -1.0*alpha)
    except ArithmeticError: # pragma: no cover
        if (abs(a) > 4.0) or (abs(b) > 2.0):
            return rational_quadratic_activation(z, min(2.5,max(-1.5,lower)),
                                                 min(3.0,max(-1.0,width)))
        elif abs(z) > NORM_EPSILON:
            return 0.0
        return 1.0
    else:
        return to_return

def mexican_hat_activation(z, lower, width):
    """
    For HyperNEAT CPPNs - see https://dx.doi.org/10.1145/2576768.2598369
    (also available from http://eplex.cs.ucf.edu/papers/risi_gecco14.pdf).
    """
    #_check_value_range(lower, 0.0, 1.0, 'mexican_hat', 'lower')
    #_check_value_range(width, 0.0, 1.0, 'mexican_hat', 'width')

    if abs(z) < (0.25+(width/2.0)):
        return 1.0
    if abs(z) == (0.25+(width/2.0)):
        return 0.875+(-0.125*lower)
    if abs(z) >= ((0.5+(width/2.0))*(lower+1.0)):
        return 0.0
    return -0.125+(-0.125*lower)

##def multiparam_hat_activation(z, lower, width, tilt):
##    _check_value_range(lower, 0.0, 1.0, 'multiparam_hat', 'lower')
##    _check_value_range(width, 0.0, 1.0, 'multiparam_hat', 'width')
##    _check_value_range(tilt, -1.0, 1.0, 'multiparam_hat', 'tilt')


##def srelu_activation(z, tilt, width):
##    _check_value_range(tilt, -1.0, 1.0, 'srelu', 'tilt')
##    _check_value_range(width, 0.0, 1.0, 'srelu', 'width')

##    gradient = 0.00001 + (0.99998*((tilt+1.0)/2.0))
##    threshold_low = 0.499 - (width*0.498*2.0)
##    threshold_high = 1.0 - threshold_low

##    if z <= threshold_low:
##        return threshold_low + ((z - threshold_low)*gradient)
##    elif z >= threshold_high:
##        return threshold_high + ((z - threshold_high)*gradient)
##    else:
##        return z

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
        self.add('sigmoid_approx', sigmoid_approx_activation)
        self.add('tanh_approx', tanh_approx_activation)
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

        self.add_shared_name('tilt',
                             **{'min_value':-1.0,
                                'max_value':1.0})
        self.add_shared_name('lower',
                             **{'min_value':0.0,
                                'max_value':1.0})
        self.add_shared_name('curve',
                             **{'min_value':0.0,
                                'max_value':1.0})
        self.add_shared_name('width',
                             **{'min_value':0.0,
                                'max_value':1.0})
        self.add_shared_name('g_curve',
                             **{'init_mean':-0.239897,
                                'init_stdev':0.8660254037844387,
                                'min_value':-12.0, 'max_value':12.0,
                                'init_type':'gaussian'})

        self.add('multiparam_relu', multiparam_relu_activation) # + tilt
##        self.add('multiparam_elu', multiparam_elu_activation_inner,
##                 a={'min_value':-1.0, 'max_value':1.0},
##                 b={'min_value':-1.0, 'max_value':1.0})
        self.add('multiparam_elu', multiparam_elu_activation) # + tilt, lower
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
        self.add('weighted_lu', weighted_lu_activation) # + tilt, curve
        self.add('multiparam_relu_softplus', multiparam_relu_softplus_activation) # + tilt, curve
        self.add('clamped_tanh_step', clamped_tanh_step_activation) # + tilt
        self.add('clamped_step', clamped_step_activation) # + tilt
##        self.add('extended_clamped_step', extended_clamped_step_activation,
##                 tilt2={'min_value':-1.0, 'max_value': 1.5})
        self.add('multiparam_sigmoid', multiparam_sigmoid_activation) # + tilt
        self.add('hat_gauss_rectangular', hat_gauss_rectangular_activation) # + curve, width
        self.add('scaled_expanded_log', scaled_expanded_log_activation) # + tilt
        self.add('multiparam_log_inv', multiparam_log_inv_activation) # + tilt
        self.add('scaled_log1p', scaled_log1p_activation) # + tilt
        self.add('multiparam_tanh_log1p', multiparam_tanh_log1p_activation) # + tilt, curve
        self.add('clamped_log1p_step', clamped_log1p_step_activation) # + tilt, curve
        self.add('multiparam_pow', multiparam_pow_activation) # + tilt
##                 a={'min_init_value':-1.0, 'max_init_value':4.0,
##                    'min_value':-3.0, 'max_value':16.0,
##                    'init_type':'gaussian'})
##        self.add('multiparam_exp', multiparam_exp_activation,
##                 a={'min_value':0.0, 'max_value':1.0})
        self.add('wave', wave_activation) # + width
        self.add('multiparam_tanh_approx', multiparam_tanh_approx_activation) # + tilt, g_curve
        self.add('multiparam_sigmoid_approx', multiparam_sigmoid_approx_activation) # + g_curve
        self.add('multiparam_gauss', multiparam_gauss_activation) # + lower, width
##        self.add('bisigmoid', bisigmoid_activation) # + tilt, width
##        self.add('bicentral', bicentral_activation) # + tilt, lower
        self.add('bicentral', bicentral_activation) # + tilt, lower, width
        self.add('multiparam_softplus', multiparam_softplus_activation) # + curve
        self.add('fourth_square_abs', fourth_square_abs_activation) # + width
        self.add('rational_quadratic', rational_quadratic_activation) # + lower, width
        self.add('mexican_hat', mexican_hat_activation) # + lower, width
##        self.add('srelu', srelu_activation) # + tilt, width


    def add(self, name, function, **kwargs):
        self.multiparameterset.add_func(name, function, 'activation', **kwargs)

    def add_shared_name(self, name, **kwargs):
        self.multiparameterset.add_shared_name(name, 'activation', **kwargs)

    def get(self, name):
        return self.multiparameterset.get_func(name, 'activation')

    def __getitem__(self, index):
        warnings.warn(
            "Use get, not indexing ([{!s}]), for activation functions".format(saferepr(index)),
            DeprecationWarning)
        return self.get(index)

    def is_valid(self, name):
        return self.multiparameterset.is_valid_func(name, 'activation')
