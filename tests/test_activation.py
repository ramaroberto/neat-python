from __future__ import print_function, division
import os
import math
import sys
import warnings

from argparse import Namespace

import neat
from neat import activations, multiparameter, repr_util

warnings.simplefilter('default')

# TODO: These tests are just smoke tests to make sure nothing has become badly broken.
# Expand to include more detailed tests of actual functionality.

class NotAlmostEqualException(AssertionError):
    pass


def assert_almost_equal(a, b):
    if abs(a - b) > 1e-6:
        max_abs = max(abs(a), abs(b))
        abs_rel_err = abs(a - b) / max_abs
        if abs_rel_err > 1e-6:
            raise NotAlmostEqualException("{0!r} !~= {1!r}".format(float(a), float(b)))

def assert_inv_func_adds_to(f, a, b):
    if isinstance(a, (float,int)):
        if b:
            assert_almost_equal((f(a)+f(-a)),b)
        else:
            assert_almost_equal(f(a),-1*f(-1*a))
    elif isinstance(a, (list,tuple,set)):
        for n in list(a):
            if b:
                assert_almost_equal((f(n)+f(-n)),b)
            else:
                assert_almost_equal(f(n),-1*f(-1*n))
    else:
        raise RuntimeError(
            "Don't know what to do with type {0!s} parameter {0!r}".format(
                type(a),a))

def test_sigmoid():
    assert activations.sigmoid_activation(0.0) == 0.5
    assert_inv_func_adds_to(activations.sigmoid_activation,
                            [0.5,1.0],1.0)
    assert_almost_equal(activations.sigmoid_activation(sys.float_info.max),1.0)
    assert_almost_equal(activations.sigmoid_activation(-1*sys.float_info.max),0.0)

def test_tanh():
    assert activations.tanh_activation(0.0) == 0.0
    assert_inv_func_adds_to(activations.tanh_activation,
                            [0.25,0.5,0.75,1.0],0.0)
    assert_almost_equal(activations.tanh_activation(sys.float_info.max),1.0)
    assert_almost_equal(activations.tanh_activation(-1*sys.float_info.max),-1.0)

def test_sigmoid_approx():
    assert activations.sigmoid_approx_activation(0.0) == 0.5
    assert_inv_func_adds_to(activations.sigmoid_approx_activation,
                            [0.5,1.0],1.0)
    assert_almost_equal(activations.sigmoid_approx_activation(sys.float_info.max),1.0)
    assert_almost_equal(activations.sigmoid_approx_activation(-1*sys.float_info.max),0.0)

def test_tanh_approx():
    assert activations.tanh_approx_activation(0.0) == 0.0
    assert_inv_func_adds_to(activations.tanh_approx_activation,
                            [0.25,0.5,0.75,1.0],0.0)
    assert_almost_equal(activations.tanh_approx_activation(sys.float_info.max),1.0)
    assert_almost_equal(activations.tanh_approx_activation(-1*sys.float_info.max),-1.0)

def test_sin():
    assert activations.sin_activation(0.0) == 0.0
    assert_inv_func_adds_to(activations.sin_activation,
                            [0.25,0.5,0.75,1.0],0.0)

def test_gauss():
    assert_almost_equal(activations.gauss_activation(0.0), 1.0)
    assert_almost_equal(activations.gauss_activation(-1.0),
                        activations.gauss_activation(1.0))
    assert_almost_equal(activations.gauss_activation(-0.5),
                        activations.gauss_activation(0.5))
    assert_almost_equal(activations.gauss_activation(sys.float_info.max),0.0)


def test_relu():
    assert activations.relu_activation(-1.0) == 0.0
    assert activations.relu_activation(0.0) == 0.0
    assert activations.relu_activation(1.0) == 1.0


def test_softplus():
    assert_almost_equal(activations.softplus_activation(-5.0),0.0)
    assert_almost_equal((activations.softplus_activation(1.0)
                         -activations.softplus_activation(-1.0)),1.0)
    assert_almost_equal(activations.softplus_activation(5.0),5.0)
    assert_almost_equal(activations.softplus_activation(-0.5),(108/6845))
    assert_almost_equal(activations.softplus_activation(0.0),(1321/9529))
    assert_almost_equal(activations.softplus_activation(0.25),(1635/5443))


def test_identity():
    assert activations.identity_activation(-1.0) == -1.0
    assert activations.identity_activation(0.0) == 0.0
    assert activations.identity_activation(1.0) == 1.0


def test_clamped():
    assert activations.clamped_activation(-2.0) == -1.0
    assert activations.clamped_activation(-1.0) == -1.0
    assert activations.clamped_activation(0.0) == 0.0
    assert activations.clamped_activation(1.0) == 1.0
    assert activations.clamped_activation(2.0) == 1.0

def test_inv():
    assert activations.inv_activation(-1.0) == -1.0
    assert activations.inv_activation(-0.5) == -2.0
    assert activations.inv_activation(0.0) == 0.0
    assert activations.inv_activation(0.5) == 2.0
    assert activations.inv_activation(1.0) == 1.0

def test_log():
    assert_almost_equal(activations.log_activation(0.75),(-1817/6316))
    assert_almost_equal(activations.log_activation(0.5),(-4319/6231))
    assert_almost_equal(activations.log_activation(0.25),(-11369/8201))
    assert activations.log_activation(1.0) == 0.0
    for i in [0.25,0.5,0.75]:
        assert_almost_equal(activations.log_activation(math.exp(i)),i)

def test_expanded_log():
    assert activations.expanded_log_activation(-1.0) == -1.0
    assert activations.expanded_log_activation(0.0) <= -6.5
    assert activations.expanded_log_activation(0.5) == 0.0
    assert activations.expanded_log_activation(1.0) == 1.0
    assert_almost_equal(activations.expanded_log_activation(-0.75),
                        -1*activations.expanded_log_activation(0.75))

def test_skewed_log1p():
    assert_inv_func_adds_to(activations.skewed_log1p_activation,
                            [0.25,0.5,0.75,1.0],0.0)
    assert activations.skewed_log1p_activation(0.0) == -1.0

def test_log1p():
    assert_inv_func_adds_to(activations.log1p_activation,
                            [0.25,0.5,0.75,1.0],0.0)
    assert activations.log1p_activation(0.0) == 0.0

def test_exp():
    for i in [0.25,0.5,0.75]:
        assert_almost_equal(math.log(activations.exp_activation(-i)),-i)
        assert_almost_equal(math.log(activations.exp_activation(i)),i)
    assert activations.exp_activation(0.0) == 1.0
    assert_almost_equal(activations.exp_activation(-0.75),(1000/2117))
    assert_almost_equal(activations.exp_activation(-0.25),(6767/8689))
    assert_almost_equal(activations.exp_activation(0.25),(8689/6767))
    assert_almost_equal(activations.exp_activation(1.0),(25946/9545))



def test_abs():
    assert activations.abs_activation(-1.0) == 1.0
    assert activations.abs_activation(0.0) == 0.0
    assert activations.abs_activation(-1.0) == 1.0


def test_hat():
    assert activations.hat_activation(-1.0) == 0.0
    assert activations.hat_activation(-0.5) == 0.5
    assert activations.hat_activation(0.0) == 1.0
    assert activations.hat_activation(0.5) == 0.5
    assert activations.hat_activation(1.0) == 0.0


def test_square():
    assert activations.square_activation(-1.0) == 1.0
    assert activations.square_activation(-0.5) == 0.25
    assert activations.square_activation(0.0) == 0.0
    assert activations.square_activation(0.5) == 0.25
    assert activations.square_activation(1.0) == 1.0


def test_cube():
    assert activations.cube_activation(-1.0) == -1.0
    assert activations.cube_activation(-0.5) == -0.125
    assert activations.cube_activation(0.0) == 0.0
    assert activations.cube_activation(0.5) == 0.125
    assert activations.cube_activation(1.0) == 1.0

def test_square_wave():
    assert activations.square_wave_activation(-1.0) == 1.0
    assert activations.square_wave_activation(-0.75) == 1.0
    assert activations.square_wave_activation(-0.5) == -1.0
    assert activations.square_wave_activation(-0.25) == -1.0
    assert activations.square_wave_activation(0.0) == 0.0
    assert activations.square_wave_activation(0.25) == 1.0
    assert activations.square_wave_activation(0.5) == 1.0
    assert activations.square_wave_activation(0.75) == -1.0
    assert activations.square_wave_activation(1.0) == -1.0

def test_triangle_wave():
    assert activations.triangle_wave_activation(0.0) == 0.0
    assert_almost_equal(activations.triangle_wave_activation(-0.75),
                        -1*activations.triangle_wave_activation(0.75))
    assert_almost_equal(activations.triangle_wave_activation(-0.5),
                        -1*activations.triangle_wave_activation(0.5))
    assert_almost_equal(activations.triangle_wave_activation(-0.25),
                        -1*activations.triangle_wave_activation(0.25))
    assert_almost_equal(activations.triangle_wave_activation(-1.0),
                        -1*activations.triangle_wave_activation(1.0))

def test_rectangular():
    assert activations.rectangular_activation(-1.0) == 0.0
    assert activations.rectangular_activation(-0.75) == 0.0
    assert activations.rectangular_activation(-0.5) == 0.5
    assert activations.rectangular_activation(-0.25) == 1.0
    assert activations.rectangular_activation(0.0) == 1.0
    assert activations.rectangular_activation(0.25) == 1.0
    assert activations.rectangular_activation(0.5) == 0.5
    assert activations.rectangular_activation(0.75) == 0.0
    assert activations.rectangular_activation(1.0) == 0.0

def test_multiparam_relu():
    assert activations.multiparam_relu_activation(-1.0,1.0) == -1.0
    assert activations.multiparam_relu_activation(0.0,1.0) == 0.0
    assert activations.multiparam_relu_activation(1.0,1.0) == 1.0
    assert activations.multiparam_relu_activation(-1.0,0.5) == -0.5
    assert activations.multiparam_relu_activation(0.0,0.5) == 0.0
    assert activations.multiparam_relu_activation(1.0,0.5) == 1.0
    assert activations.multiparam_relu_activation(0.0,0.0) == 0.0
    assert activations.multiparam_relu_activation(0.5,0.0) == 0.5
    assert activations.multiparam_relu_activation(1.0,0.0) == 1.0
    assert activations.multiparam_relu_activation(-1.0,-0.5) == 0.5
    assert activations.multiparam_relu_activation(0.0,-0.5) == 0.0
    assert activations.multiparam_relu_activation(1.0,-0.5) == 1.0
    assert activations.multiparam_relu_activation(-1.0,-1.0) == 1.0
    assert activations.multiparam_relu_activation(0.0,-1.0) == 0.0
    assert activations.multiparam_relu_activation(1.0,-1.0) == 1.0

def test_multiparam_elu():
    assert_almost_equal(activations.multiparam_elu_activation(-1.0,1.0,0.0),(-610/8347))
    assert_almost_equal(activations.multiparam_elu_activation(-0.5,-1.0,0.0),(-601/4997))
    assert_almost_equal(activations.multiparam_elu_activation(-1.0,1.0,0.25),(-424/3519))
    assert_almost_equal(activations.multiparam_elu_activation(-1.0,0.0,0.25),(-877/4607))
    assert_almost_equal(activations.multiparam_elu_activation(-1.0,1.0,0.5),(-855/4304))
    assert_almost_equal(activations.multiparam_elu_activation(-1.0,0.0,0.5),(-1477/4706))
    assert_almost_equal(activations.multiparam_elu_activation(-1.0,1.0,0.75),(-620/1893))
    assert_almost_equal(activations.multiparam_elu_activation(-1.0,-0.5,0.5),(-3504/9137))
    assert_almost_equal(activations.multiparam_elu_activation(-1.0,0.0,0.75),(-3364/6501))
    assert_almost_equal(activations.multiparam_elu_activation(-1.0,1.0,1.0),(-1600/2963))
    assert_almost_equal(activations.multiparam_elu_activation(-1.0,-0.5,1.0),(-1000/2117))
    assert activations.multiparam_elu_activation(0.0,1.0,1.0) == 0.0
    assert activations.multiparam_elu_activation(0.5,1.0,1.0) == 0.5
    assert activations.multiparam_elu_activation(1.0,1.0,1.0) == 1.0
    assert activations.multiparam_elu_activation(0.0,1.0,0.75) == 0.0
    assert activations.multiparam_elu_activation(1.0,1.0,0.75) == 1.0
    assert activations.multiparam_elu_activation(0.0,1.0,0.5) == 0.0
    assert activations.multiparam_elu_activation(0.5,1.0,0.5) == 0.5
    assert activations.multiparam_elu_activation(1.0,1.0,0.5) == 1.0
    assert activations.multiparam_elu_activation(0.0,1.0,0.25) == 0.0
    assert activations.multiparam_elu_activation(1.0,1.0,0.25) == 1.0
    assert activations.multiparam_elu_activation(0.0,1.0,0.0) == 0.0
    assert activations.multiparam_elu_activation(0.5,1.0,0.0) == 0.5
    assert activations.multiparam_elu_activation(1.0,1.0,0.0) == 1.0
    assert activations.multiparam_elu_activation(0.0,0.5,1.0) == 0.0
    assert activations.multiparam_elu_activation(0.0,0.5,0.5) == 0.0
    assert activations.multiparam_elu_activation(0.0,0.5,0.0) == 0.0
    assert activations.multiparam_elu_activation(0.0,0.0,1.0) == 0.0
    assert activations.multiparam_elu_activation(0.0,0.0,0.75) == 0.0
    assert activations.multiparam_elu_activation(0.0,0.0,0.5) == 0.0
    assert activations.multiparam_elu_activation(0.0,0.0,0.25) == 0.0
    assert activations.multiparam_elu_activation(0.0,0.0,0.0) == 0.0
    assert activations.multiparam_elu_activation(0.0,-0.5,1.0) == 0.0
    assert activations.multiparam_elu_activation(0.0,-0.5,0.5) == 0.0
    assert activations.multiparam_elu_activation(0.0,-0.5,0.0) == 0.0
    assert activations.multiparam_elu_activation(0.0,-1.0,1.0) == 0.0
    assert activations.multiparam_elu_activation(0.0,-1.0,0.75) == 0.0
    assert activations.multiparam_elu_activation(0.0,-1.0,0.5) == 0.0
    assert activations.multiparam_elu_activation(0.0,-1.0,0.25) == 0.0
    assert activations.multiparam_elu_activation(0.0,-1.0,0.0) == 0.0
    assert_almost_equal(activations.multiparam_elu_activation(-0.5,-1.0,1.0),
                        activations.multiparam_elu_activation(-0.5,-1.0,0.5))
    assert_almost_equal(activations.multiparam_elu_activation(-1.0,-1.0,1.0),
                        activations.multiparam_elu_activation(-1.0,-1.0,0.5))
    assert_almost_equal(activations.multiparam_elu_activation(0.5,-1.0,1.0),
                        activations.multiparam_elu_activation(0.5,-1.0,0.0))
    assert_almost_equal(activations.multiparam_elu_activation(0.5,0.0,1.0),
                        activations.multiparam_elu_activation(0.5,0.0,0.0))
    assert_almost_equal(activations.multiparam_elu_activation(1.0,-1.0,1.0),
                        activations.multiparam_elu_activation(1.0,-1.0,0.0))
    assert_almost_equal(activations.multiparam_elu_activation(1.0,-0.5,1.0),
                        activations.multiparam_elu_activation(1.0,-0.5,0.0))
    assert_almost_equal(activations.multiparam_elu_activation(1.0,0.0,1.0),
                        activations.multiparam_elu_activation(1.0,0.0,0.0))
    assert_almost_equal(activations.multiparam_elu_activation(1.0,0.5,1.0),
                        activations.multiparam_elu_activation(1.0,0.5,0.0))
    assert_almost_equal(activations.multiparam_elu_activation(-60.0,0.5,0.5),
                        activations.multiparam_elu_activation((-1*sys.float_info.max/5),
                                                              0.5,0.5))
    assert_almost_equal(activations.multiparam_elu_activation(-0.5,sys.float_info.max_10_exp,0.5),
                        activations.multiparam_elu_activation(-0.5,sys.float_info.max,0.5))
    assert_almost_equal(activations.multiparam_elu_activation(-0.5,0.5,sys.float_info.max_10_exp),
                        activations.multiparam_elu_activation(-0.5,0.5,sys.float_info.max))


def test_weighted_lu():
    assert activations.weighted_lu_activation(0.0,1.0,1.0) == 0.0
    assert activations.weighted_lu_activation(0.5,1.0,1.0) == 0.5
    assert activations.weighted_lu_activation(1.0,1.0,1.0) == 1.0
    assert activations.weighted_lu_activation(0.0,1.0,0.5) == 0.0
    assert activations.weighted_lu_activation(1.0,1.0,0.5) == 1.0
    assert activations.weighted_lu_activation(0.0,1.0,0.0) == 0.0
    assert activations.weighted_lu_activation(0.5,1.0,0.0) == 0.5
    assert activations.weighted_lu_activation(1.0,1.0,0.0) == 1.0
    assert activations.weighted_lu_activation(0.0,1.0,-0.5) == 0.0
    assert activations.weighted_lu_activation(1.0,1.0,-0.5) == 1.0
    assert activations.weighted_lu_activation(0.0,1.0,-1.0) == 0.0
    assert activations.weighted_lu_activation(0.5,1.0,-1.0) == 0.5
    assert activations.weighted_lu_activation(1.0,1.0,-1.0) == 1.0
    assert activations.weighted_lu_activation(0.0,0.75,1.0) == 0.0
    assert activations.weighted_lu_activation(1.0,0.75,1.0) == 1.0
    assert activations.weighted_lu_activation(0.0,0.75,0.0) == 0.0
    assert activations.weighted_lu_activation(1.0,0.75,0.0) == 1.0
    assert activations.weighted_lu_activation(0.0,0.75,-1.0) == 0.0
    assert activations.weighted_lu_activation(1.0,0.75,-1.0) == 1.0
    assert activations.weighted_lu_activation(0.0,0.5,1.0) == 0.0
    assert activations.weighted_lu_activation(0.5,0.5,1.0) == 0.5
    assert activations.weighted_lu_activation(1.0,0.5,1.0) == 1.0
    assert activations.weighted_lu_activation(0.0,0.5,0.5) == 0.0
    assert activations.weighted_lu_activation(1.0,0.5,0.5) == 1.0
    assert activations.weighted_lu_activation(0.0,0.5,0.0) == 0.0
    assert activations.weighted_lu_activation(0.5,0.5,0.0) == 0.5
    assert activations.weighted_lu_activation(1.0,0.5,0.0) == 1.0
    assert activations.weighted_lu_activation(0.0,0.5,-0.5) == 0.0
    assert activations.weighted_lu_activation(1.0,0.5,-0.5) == 1.0
    assert activations.weighted_lu_activation(0.0,0.5,-1.0) == 0.0
    assert activations.weighted_lu_activation(0.5,0.5,-1.0) == 0.5
    assert activations.weighted_lu_activation(1.0,0.5,-1.0) == 1.0
    assert activations.weighted_lu_activation(0.0,0.25,1.0) == 0.0
    assert activations.weighted_lu_activation(1.0,0.25,1.0) == 1.0
    assert activations.weighted_lu_activation(0.0,0.25,0.0) == 0.0
    assert activations.weighted_lu_activation(1.0,0.25,0.0) == 1.0
    assert activations.weighted_lu_activation(0.0,0.25,-1.0) == 0.0
    assert activations.weighted_lu_activation(1.0,0.25,-1.0) == 1.0
    assert activations.weighted_lu_activation(-1.0,0.0,1.0) == -1.0
    assert activations.weighted_lu_activation(-0.5,0.0,1.0) == -0.5
    assert activations.weighted_lu_activation(0.0,0.0,1.0) == 0.0
    assert activations.weighted_lu_activation(0.5,0.0,1.0) == 0.5
    assert activations.weighted_lu_activation(1.0,0.0,1.0) == 1.0
    assert activations.weighted_lu_activation(-1.0,0.0,0.5) == -0.5
    assert activations.weighted_lu_activation(0.0,0.0,0.5) == 0.0
    assert activations.weighted_lu_activation(1.0,0.0,0.5) == 1.0
    assert activations.weighted_lu_activation(0.0,0.0,0.0) == 0.0
    assert activations.weighted_lu_activation(0.5,0.0,0.0) == 0.5
    assert activations.weighted_lu_activation(1.0,0.0,0.0) == 1.0
    assert activations.weighted_lu_activation(-1.0,0.0,-0.5) == 0.5
    assert activations.weighted_lu_activation(0.0,0.0,-0.5) == 0.0
    assert activations.weighted_lu_activation(1.0,0.0,-0.5) == 1.0
    assert activations.weighted_lu_activation(-1.0,0.0,-1.0) == 1.0
    assert activations.weighted_lu_activation(-0.5,0.0,-1.0) == 0.5
    assert activations.weighted_lu_activation(0.0,0.0,-1.0) == 0.0
    assert activations.weighted_lu_activation(0.5,0.0,-1.0) == 0.5
    assert activations.weighted_lu_activation(1.0,0.0,-1.0) == 1.0
    assert_almost_equal(activations.weighted_lu_activation(-1.0,1.0,-1.0),(-610/8347))
    assert_almost_equal(activations.weighted_lu_activation(-1.0,0.5,0.0),(-855/8608))
    assert_almost_equal(activations.weighted_lu_activation(-1.0,1.0,-0.5),(-424/3519))
    assert_almost_equal(activations.weighted_lu_activation(-1.0,1.0,0.0),(-855/4304))
    assert_almost_equal(activations.weighted_lu_activation(-1.0,1.0,0.5),(-620/1893))
    assert_almost_equal(activations.weighted_lu_activation(-1.0,0.5,0.5),(-3133/7572))
    assert_almost_equal(activations.weighted_lu_activation(-0.5,0.5,1.0),(-3606/7219))
    assert_almost_equal(activations.weighted_lu_activation(-1.0,1.0,1.0),(-1600/2963))
    assert_almost_equal(activations.weighted_lu_activation(-1.0,0.25,-1.0),(7089/9688))
    assert_almost_equal(activations.weighted_lu_activation(-1.0,0.5,1.0),(-4563/5926))

    try:
        ignored = activations.weighted_lu_activation(-1.0,2.0,1.0)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "weighted_lu_activation(-1.0,2.0,1.0) did not raise a ValueError/derived")

def test_multiparam_softplus():
    assert 0.0 <= activations.multiparam_softplus_activation(-1.0,0.25) <= 0.000123
    assert 1.0 <= activations.multiparam_softplus_activation(1.0,0.25) <= 1.000123
    assert activations.multiparam_softplus_activation(-1.0,0.0) == 0.0
    assert activations.multiparam_softplus_activation(-0.5,0.0) == 0.0
    assert activations.multiparam_softplus_activation(0.0,0.0) == 0.0
    assert activations.multiparam_softplus_activation(0.5,0.0) == 0.5
    assert activations.multiparam_softplus_activation(1.0,0.0) == 1.0
    assert_almost_equal(activations.multiparam_softplus_activation(-0.25,0.5),(687/8348))
    assert_almost_equal(activations.multiparam_softplus_activation(0.0,0.25),(589/6594))
    assert_almost_equal(activations.multiparam_softplus_activation(-0.5,0.75),(1273/8783))
    assert_almost_equal(activations.multiparam_softplus_activation(0.0,0.5),(1243/6983))
    assert_almost_equal(activations.multiparam_softplus_activation(0.25,0.5),(1387/4174))
    assert_almost_equal(activations.multiparam_softplus_activation(-0.5,1.0),(2371/5844))
    assert_almost_equal(activations.multiparam_softplus_activation(0.0,1.0),(3793/6105))
    assert_almost_equal(activations.multiparam_softplus_activation(0.75,0.5),(4261/5581))
    assert_almost_equal(activations.multiparam_softplus_activation(0.5,1.0),(8098/8941))
    assert_almost_equal(activations.multiparam_softplus_activation(-60.0,1.0),0.0)
    for x in [0.0, 0.25, 0.5, 0.75, 1.0]:
        assert_almost_equal((activations.multiparam_softplus_activation(1.0,x)
                             -activations.multiparam_softplus_activation(-1.0,x)),1.0)
    assert_almost_equal(activations.multiparam_softplus_activation(60.0,1.0),60.0)

def test_multiparam_relu_softplus():
    assert 1.0 <= activations.multiparam_relu_softplus_activation(1.0,1.0,0.25) <= 1.000123
    assert activations.multiparam_relu_softplus_activation(-1.0,1.0,0.0) == -1.0
    assert activations.multiparam_relu_softplus_activation(-0.5,1.0,0.0) == -0.5
    assert activations.multiparam_relu_softplus_activation(0.0,1.0,0.0) == 0.0
    assert activations.multiparam_relu_softplus_activation(0.5,1.0,0.0) == 0.5
    assert activations.multiparam_relu_softplus_activation(1.0,1.0,0.0) == 1.0
    assert 1.0 <= activations.multiparam_relu_softplus_activation(1.0,0.5,0.0) <= 1.000123
    assert 0.0 <= activations.multiparam_relu_softplus_activation(-1.0,0.0,0.25) <= 0.000123
    assert 1.0 <= activations.multiparam_relu_softplus_activation(1.0,0.0,0.25) <= 1.000123
    assert 0.0 <= activations.multiparam_relu_softplus_activation(-1.0,0.0,0.0) <= 0.000123
    assert 0.0 <= activations.multiparam_relu_softplus_activation(-0.5,0.0,0.0) <= 0.000123
    assert 1.0 <= activations.multiparam_relu_softplus_activation(1.0,0.0,0.0) <= 1.000123
    assert 1.0 <= activations.multiparam_relu_softplus_activation(1.0,-0.5,0.0) <= 1.000123
    assert 1.0 <= activations.multiparam_relu_softplus_activation(1.0,-1.0,0.25) <= 1.000123
    assert activations.multiparam_relu_softplus_activation(-1.0,-1.0,0.0) == 1.0
    assert activations.multiparam_relu_softplus_activation(-0.5,-1.0,0.0) == 0.5
    assert activations.multiparam_relu_softplus_activation(0.0,-1.0,0.0) == 0.0
    assert activations.multiparam_relu_softplus_activation(0.5,-1.0,0.0) == 0.5
    assert activations.multiparam_relu_softplus_activation(1.0,-1.0,0.0) == 1.0
    assert_almost_equal(activations.multiparam_relu_softplus_activation(-1.0,0.0,0.5),(29/7469))
    assert_almost_equal(activations.multiparam_relu_softplus_activation(0.0,1.0,0.25),
                        activations.multiparam_relu_softplus_activation(0.0,-1.0,0.25))
    assert_almost_equal(activations.multiparam_relu_softplus_activation(0.0,0.5,0.0),
                        activations.multiparam_relu_softplus_activation(0.0,-0.5,0.0))
    assert_almost_equal(activations.multiparam_relu_softplus_activation(0.0,0.0,0.0),(254/9961))
    assert_almost_equal(activations.multiparam_relu_softplus_activation(0.0,1.0,0.5),
                        activations.multiparam_relu_softplus_activation(0.0,-1.0,0.5))
    assert_almost_equal(activations.multiparam_relu_softplus_activation(0.0,0.0,0.25),(449/7540))
    assert_almost_equal(activations.multiparam_relu_softplus_activation(0.0,0.5,0.5),
                        activations.multiparam_relu_softplus_activation(0.0,-0.5,0.5))
    assert_almost_equal(activations.multiparam_relu_softplus_activation(0.0,1.0,0.75),
                        activations.multiparam_relu_softplus_activation(0.0,-1.0,0.75))
    assert_almost_equal(activations.multiparam_relu_softplus_activation(0.0,0.0,0.5),(1103/8262))
    assert_almost_equal(activations.multiparam_relu_softplus_activation(-1.0,0.5,0.5),(-1601/6471))
    assert_almost_equal(activations.multiparam_relu_softplus_activation(-1.0,-0.5,0.5),(1049/4153))
    assert_almost_equal(activations.multiparam_relu_softplus_activation(0.0,0.0,0.75),(2011/6911))
    assert_almost_equal(activations.multiparam_relu_softplus_activation(0.0,1.0,1.0),
                        activations.multiparam_relu_softplus_activation(0.0,-1.0,1.0))
    assert_almost_equal(activations.multiparam_relu_softplus_activation(-1.0,-0.5,1.0),(1248/3955))
    assert_almost_equal(activations.multiparam_relu_softplus_activation(-0.5,1.0,0.5),(-3546/9677))
    assert_almost_equal(activations.multiparam_relu_softplus_activation(-0.5,-1.0,0.5),(2105/5488))
    assert_almost_equal(activations.multiparam_relu_softplus_activation(-0.5,0.0,1.0),(2371/5844))
    assert_almost_equal(activations.multiparam_relu_softplus_activation(-0.5,-1.0,1.0),(4049/8941))
    assert_almost_equal(activations.multiparam_relu_softplus_activation(0.0,0.5,1.0),
                        activations.multiparam_relu_softplus_activation(0.0,-0.5,1.0))
    assert_almost_equal(activations.multiparam_relu_softplus_activation(0.5,1.0,0.5),
                        activations.multiparam_relu_softplus_activation(0.5,-1.0,0.5))
    assert_almost_equal(activations.multiparam_relu_softplus_activation(0.5,0.0,0.5),(3018/5741))
    assert_almost_equal(activations.multiparam_relu_softplus_activation(0.0,0.0,1.0),(3793/6105))
    assert_almost_equal(activations.multiparam_relu_softplus_activation(0.5,1.0,1.0),
                        activations.multiparam_relu_softplus_activation(0.5,-1.0,1.0))
    assert_almost_equal(activations.multiparam_relu_softplus_activation(-1.0,1.0,0.5),(-5351/7147))
    assert_almost_equal(activations.multiparam_relu_softplus_activation(-1.0,-1.0,0.5),(7111/9465))
    assert_almost_equal(activations.multiparam_relu_softplus_activation(0.5,0.0,1.0),(8098/8941))
    assert_almost_equal(activations.multiparam_relu_softplus_activation(1.0,1.0,0.5),
                        activations.multiparam_relu_softplus_activation(1.0,-1.0,0.5))
    assert_almost_equal(activations.multiparam_relu_softplus_activation(1.0,0.5,0.5),
                        activations.multiparam_relu_softplus_activation(1.0,-0.5,0.5))
    assert_almost_equal(activations.multiparam_relu_softplus_activation(1.0,0.0,0.5),(7498/7469))
    assert_almost_equal(activations.multiparam_relu_softplus_activation(1.0,1.0,0.75),
                        activations.multiparam_relu_softplus_activation(1.0,-1.0,0.75))
    assert_almost_equal(activations.multiparam_relu_softplus_activation(1.0,1.0,1.0),
                        activations.multiparam_relu_softplus_activation(1.0,-1.0,1.0))
    assert_almost_equal(activations.multiparam_relu_softplus_activation(1.0,0.5,1.0),
                        activations.multiparam_relu_softplus_activation(1.0,-0.5,1.0))

def test_clamped_tanh_step():
    assert activations.clamped_tanh_step_activation(-1.0,1.0) == -1.0
    assert activations.clamped_tanh_step_activation(0.0,1.0) == 0.0
    assert activations.clamped_tanh_step_activation(1.0,1.0) == 1.0
    assert activations.clamped_tanh_step_activation(0.0,0.5) == 0.0
    assert activations.clamped_tanh_step_activation(0.0,0.0) == 0.0
    assert activations.clamped_tanh_step_activation(0.0,-0.5) == 0.0
    assert activations.clamped_tanh_step_activation(-1.0,-1.0) == -1.0
    assert activations.clamped_tanh_step_activation(0.0,-1.0) == 0.0
    assert activations.clamped_tanh_step_activation(1.0,-1.0) == 1.0
    assert_almost_equal(activations.clamped_tanh_step_activation(-0.25,0.0),
                        -1*activations.clamped_tanh_step_activation(0.25,0.0))
    assert_almost_equal(activations.clamped_tanh_step_activation(-0.5,0.5),
                        -1*activations.clamped_tanh_step_activation(0.5,0.5))
    assert_almost_equal(activations.clamped_tanh_step_activation(-0.5,0.0),
                        -1*activations.clamped_tanh_step_activation(0.5,0.0))
    assert_almost_equal(activations.clamped_tanh_step_activation(-0.5,-0.5),
                        -1*activations.clamped_tanh_step_activation(0.5,-0.5))
    assert_almost_equal(activations.clamped_tanh_step_activation(-0.75,0.0),
                        -1*activations.clamped_tanh_step_activation(0.75,0.0))
    assert_almost_equal(activations.clamped_tanh_step_activation(-1.0,0.0),
                        -1*activations.clamped_tanh_step_activation(1.0,0.0))
    assert_almost_equal(activations.clamped_tanh_step_activation(-1.0,0.5),
                        activations.clamped_tanh_step_activation(-1.0,-0.5))
    assert_almost_equal(activations.clamped_tanh_step_activation(1.0,0.5),
                        activations.clamped_tanh_step_activation(1.0,-0.5))


def test_multiparam_sigmoid():
    assert activations.multiparam_sigmoid_activation(-1.0,1.0) == 0.0
    assert activations.multiparam_sigmoid_activation(0.0,1.0) == 0.5
    assert activations.multiparam_sigmoid_activation(1.0,1.0) == 1.0
    assert_almost_equal(activations.multiparam_sigmoid_activation(-1.0,0.5),
                        activations.multiparam_sigmoid_activation(-1.0,-0.5))
    assert_almost_equal(activations.multiparam_sigmoid_activation(1.0,0.5),
                        activations.multiparam_sigmoid_activation(1.0,-0.5))
    assert activations.multiparam_sigmoid_activation(0.0,0.5) == 0.5
    assert_almost_equal((activations.multiparam_sigmoid_activation(-1.0,0.0)+
                         activations.multiparam_sigmoid_activation(1.0,0.0)),1.0)
    assert_almost_equal((activations.multiparam_sigmoid_activation(-0.5,0.0)+
                         activations.multiparam_sigmoid_activation(0.5,0.0)),1.0)
    assert activations.multiparam_sigmoid_activation(0.0,0.0) == 0.5
    assert activations.multiparam_sigmoid_activation(0.0,-0.5) == 0.5
    assert activations.multiparam_sigmoid_activation(-1.0,-1.0) == 0.0
    assert activations.multiparam_sigmoid_activation(0.0,-1.0) == 0.5
    assert activations.multiparam_sigmoid_activation(1.0,-1.0) == 1.0
    assert_almost_equal(activations.multiparam_sigmoid_activation(-1.0,0.0),(46/6873))
    assert_almost_equal(activations.multiparam_sigmoid_activation(-0.5,-0.5),(337/8885))
    assert_almost_equal(activations.multiparam_sigmoid_activation(-0.5,0.0),(674/8885))
    assert_almost_equal(activations.multiparam_sigmoid_activation(-0.25,0.0),(1605/7207))
    assert_almost_equal(activations.multiparam_sigmoid_activation(0.25,0.0),(5602/7207))
    assert_almost_equal(activations.multiparam_sigmoid_activation(0.5,0.0),(8211/8885))
    assert_almost_equal(activations.multiparam_sigmoid_activation(0.5,-0.5),(8548/8885))
    assert_almost_equal(activations.multiparam_sigmoid_activation(1.0,0.0),(6827/6873))

def test_clamped_step():
    assert activations.clamped_step_activation(-1.0,1.0) == -1.0
    assert activations.clamped_step_activation(-0.5,1.0) == -0.5
    assert activations.clamped_step_activation(0.0,1.0) == 0.0
    assert activations.clamped_step_activation(0.5,1.0) == 0.5
    assert activations.clamped_step_activation(1.0,1.0) == 1.0
    assert activations.clamped_step_activation(-1.0,0.5) == -1.0
    assert activations.clamped_step_activation(0.0,0.5) == 0.0
    assert activations.clamped_step_activation(1.0,0.5) == 1.0
    assert activations.clamped_step_activation(-1.0,0.0) == -1.0
    assert activations.clamped_step_activation(-0.75,0.0) == -1.0
    assert activations.clamped_step_activation(-0.5,0.0) == -1.0
    assert activations.clamped_step_activation(-0.25,0.0) == -0.5
    assert activations.clamped_step_activation(0.0,0.0) == 0.0
    assert activations.clamped_step_activation(0.25,0.0) == 0.5
    assert activations.clamped_step_activation(0.5,0.0) == 1.0
    assert activations.clamped_step_activation(0.75,0.0) == 1.0
    assert activations.clamped_step_activation(1.0,0.0) == 1.0
    assert activations.clamped_step_activation(-1.0,-0.5) == -1.0
    assert activations.clamped_step_activation(-0.5,-0.5) == -1.0
    assert activations.clamped_step_activation(0.0,-0.5) == 0.0
    assert activations.clamped_step_activation(0.5,-0.5) == 1.0
    assert activations.clamped_step_activation(1.0,-0.5) == 1.0
    assert activations.clamped_step_activation(-1.0,-1.0) == -1.0
    assert activations.clamped_step_activation(-0.5,-1.0) == -1.0
    assert activations.clamped_step_activation(0.0,-1.0) == 0.0
    assert activations.clamped_step_activation(0.5,-1.0) == 1.0
    assert activations.clamped_step_activation(1.0,-1.0) == 1.0
    assert_almost_equal(activations.clamped_step_activation(-0.5,0.5),
                        -1*activations.clamped_step_activation(0.5,0.5))


def test_hat_gauss_rectangular():
    assert activations.hat_gauss_rectangular_activation(0.0,1.0,1.0) == 1.0
    assert activations.hat_gauss_rectangular_activation(0.0,1.0,0.75) == 1.0
    assert activations.hat_gauss_rectangular_activation(0.0,1.0,0.5) == 1.0
    assert activations.hat_gauss_rectangular_activation(0.0,1.0,0.25) == 1.0
    assert activations.hat_gauss_rectangular_activation(-1.0,1.0,0.0) == 0.0
    assert activations.hat_gauss_rectangular_activation(-0.5,1.0,0.0) == 0.5
    assert activations.hat_gauss_rectangular_activation(0.0,1.0,0.0) == 1.0
    assert activations.hat_gauss_rectangular_activation(0.5,1.0,0.0) == 0.5
    assert activations.hat_gauss_rectangular_activation(1.0,1.0,0.0) == 0.0
    assert activations.hat_gauss_rectangular_activation(0.0,0.75,1.0) == 1.0
    assert activations.hat_gauss_rectangular_activation(0.0,0.75,0.5) == 1.0
    assert activations.hat_gauss_rectangular_activation(-1.0,0.75,0.0) == 0.0
    assert activations.hat_gauss_rectangular_activation(0.0,0.75,0.0) == 1.0
    assert activations.hat_gauss_rectangular_activation(1.0,0.75,0.0) == 0.0
    assert activations.hat_gauss_rectangular_activation(0.0,0.5,1.0) == 1.0
    assert activations.hat_gauss_rectangular_activation(0.0,0.5,0.75) == 1.0
    assert activations.hat_gauss_rectangular_activation(0.0,0.5,0.5) == 1.0
    assert activations.hat_gauss_rectangular_activation(0.0,0.5,0.25) == 1.0
    assert activations.hat_gauss_rectangular_activation(-1.0,0.5,0.0) == 0.0
    assert activations.hat_gauss_rectangular_activation(-0.5,0.5,0.0) == 0.5
    assert activations.hat_gauss_rectangular_activation(0.0,0.5,0.0) == 1.0
    assert activations.hat_gauss_rectangular_activation(0.5,0.5,0.0) == 0.5
    assert activations.hat_gauss_rectangular_activation(1.0,0.5,0.0) == 0.0
    assert activations.hat_gauss_rectangular_activation(0.0,0.25,1.0) == 1.0
    assert activations.hat_gauss_rectangular_activation(0.0,0.25,0.5) == 1.0
    assert activations.hat_gauss_rectangular_activation(-1.0,0.25,0.0) == 0.0
    assert activations.hat_gauss_rectangular_activation(0.0,0.25,0.0) == 1.0
    assert activations.hat_gauss_rectangular_activation(1.0,0.25,0.0) == 0.0
    assert activations.hat_gauss_rectangular_activation(0.0,0.0,1.0) == 1.0
    assert activations.hat_gauss_rectangular_activation(0.0,0.0,0.75) == 1.0
    assert activations.hat_gauss_rectangular_activation(0.0,0.0,0.5) == 1.0
    assert activations.hat_gauss_rectangular_activation(0.0,0.0,0.25) == 1.0
    assert activations.hat_gauss_rectangular_activation(-1.0,0.0,0.0) == 0.0
    assert activations.hat_gauss_rectangular_activation(-0.5,0.0,0.0) == 0.5
    assert activations.hat_gauss_rectangular_activation(0.0,0.0,0.0) == 1.0
    assert activations.hat_gauss_rectangular_activation(0.5,0.0,0.0) == 0.5
    assert activations.hat_gauss_rectangular_activation(1.0,0.0,0.0) == 0.0
    for z in [0.25, 0.5, 0.75, 1.0]:
        for a in [0.0, 0.5, 1.0]:
            for b in [0.0, 0.25, 0.5, 0.75, 1.0]:
                assert_almost_equal(activations.hat_gauss_rectangular_activation(z,a,b),
                                    activations.hat_gauss_rectangular_activation(-z,a,b))
    assert_almost_equal(activations.hat_gauss_rectangular_activation(-1.0,1.0,0.25),
                        activations.hat_gauss_rectangular_activation(1.0,0.5,0.25))
    assert_almost_equal(activations.hat_gauss_rectangular_activation(-1.0,1.0,0.5),
                        activations.hat_gauss_rectangular_activation(1.0,0.25,0.5))
    assert_almost_equal(activations.hat_gauss_rectangular_activation(-1.0,1.0,0.75),
                        activations.hat_gauss_rectangular_activation(1.0,0.5,0.75))
    assert_almost_equal(activations.hat_gauss_rectangular_activation(-1.0,1.0,1.0),
                        activations.hat_gauss_rectangular_activation(1.0,0.25,1.0))


def test_scaled_expanded_log():
    assert activations.scaled_expanded_log_activation(-0.5,1.0) == 2.0
    assert activations.scaled_expanded_log_activation(0.0,1.0) <= -13.0
    assert activations.scaled_expanded_log_activation(0.5,1.0) == -2.0
    assert activations.scaled_expanded_log_activation(1.0,1.0) == 0.0
    assert activations.scaled_expanded_log_activation(-1.0,0.0) == -1.0
    assert activations.scaled_expanded_log_activation(-0.25,0.0) == 1.0
    assert activations.scaled_expanded_log_activation(0.0,0.0) <= -6.5
    assert activations.scaled_expanded_log_activation(0.25,0.0) == -1.0
    assert activations.scaled_expanded_log_activation(0.5,0.0) == 0.0
    assert activations.scaled_expanded_log_activation(1.0,0.0) == 1.0
    assert activations.scaled_expanded_log_activation(-1.0,-1.0) == -1.0
    assert activations.scaled_expanded_log_activation(-0.5,-1.0) == -0.5
    assert activations.scaled_expanded_log_activation(0.0,-1.0) <= -3.25
    assert activations.scaled_expanded_log_activation(0.5,-1.0) == 0.5
    assert activations.scaled_expanded_log_activation(1.0,-1.0) == 1.0
    assert_almost_equal(activations.scaled_expanded_log_activation(-0.5,-0.5),
                        -1*activations.scaled_expanded_log_activation(0.5,-0.5))
    assert_almost_equal(activations.scaled_expanded_log_activation(-0.75,0.0),
                        -1*activations.scaled_expanded_log_activation(0.75,0.0))
    assert_almost_equal(activations.scaled_expanded_log_activation(-0.5,0.5),
                        -1*activations.scaled_expanded_log_activation(0.5,0.5))
    assert_almost_equal(activations.scaled_expanded_log_activation(-1.0,0.5),
                        -1*activations.scaled_expanded_log_activation(1.0,0.5))
    assert_almost_equal(activations.scaled_expanded_log_activation(-1.0,-0.5),
                        -1*activations.scaled_expanded_log_activation(1.0,-0.5))

def test_multiparam_log_inv():
    assert activations.multiparam_log_inv_activation(-0.5,1.0) == 2.0
    assert activations.multiparam_log_inv_activation(0.0,1.0) <= -13.0
    assert activations.multiparam_log_inv_activation(0.5,1.0) == -2.0
    assert activations.multiparam_log_inv_activation(1.0,1.0) == 0.0
    assert activations.multiparam_log_inv_activation(-1.0,0.0) == -1.0
    assert activations.multiparam_log_inv_activation(-0.25,0.0) == 1.0
    assert activations.multiparam_log_inv_activation(0.0,0.0) <= -6.5
    assert activations.multiparam_log_inv_activation(0.25,0.0) == -1.0
    assert activations.multiparam_log_inv_activation(0.5,0.0) == 0.0
    assert activations.multiparam_log_inv_activation(1.0,0.0) == 1.0
    assert activations.multiparam_log_inv_activation(-1.0,-0.5) == 0.0
    assert activations.multiparam_log_inv_activation(-0.5,-0.5) == 1.0
    assert activations.multiparam_log_inv_activation(0.0,-0.5) <= -3.25
    assert activations.multiparam_log_inv_activation(0.5,-0.5) == -1.0
    assert activations.multiparam_log_inv_activation(1.0,-0.5) == 0.0
    assert activations.multiparam_log_inv_activation(-1.0,-1.0) == 1.0
    assert activations.multiparam_log_inv_activation(-0.5,-1.0) == 2.0
    assert activations.multiparam_log_inv_activation(0.0,-1.0) == 0.0
    assert activations.multiparam_log_inv_activation(0.5,-1.0) == -2.0
    assert activations.multiparam_log_inv_activation(1.0,-1.0) == -1.0
    assert_almost_equal(activations.multiparam_log_inv_activation(-0.75,0.0),
                        -1*activations.multiparam_log_inv_activation(0.75,0.0))
    assert_almost_equal(activations.multiparam_log_inv_activation(-0.5,0.5),
                        -1*activations.multiparam_log_inv_activation(0.5,0.5))
    assert_almost_equal(activations.multiparam_log_inv_activation(-1.0,0.5),
                        -1*activations.multiparam_log_inv_activation(1.0,0.5))

def test_scaled_log1p():
    for n in [0.5,1.0]:
        for m in [0.0,0.5,1.0,1.5,2.0]:
            assert_almost_equal(activations.scaled_log1p_activation(n,m),
                                -1*activations.scaled_log1p_activation(-1*n,m))
    assert activations.scaled_log1p_activation(0.0,1.0) == 0.0
    assert activations.scaled_log1p_activation(0.0,0.5) == 0.0
    assert activations.scaled_log1p_activation(0.0,0.0) == 0.0
    assert activations.scaled_log1p_activation(0.0,-0.5) == 0.0
    assert activations.scaled_log1p_activation(0.0,-1.0) == 0.0

def test_multiparam_tanh_log1p():
    assert activations.multiparam_tanh_log1p_activation(-1.0,1.0,1.0) == -1.0
    assert activations.multiparam_tanh_log1p_activation(-0.5,1.0,1.0) == -0.5
    assert activations.multiparam_tanh_log1p_activation(0.0,1.0,1.0) == 0.0
    assert activations.multiparam_tanh_log1p_activation(0.5,1.0,1.0) == 0.5
    assert activations.multiparam_tanh_log1p_activation(1.0,1.0,1.0) == 1.0
    assert activations.multiparam_tanh_log1p_activation(0.0,1.0,0.5) == 0.0
    assert activations.multiparam_tanh_log1p_activation(0.0,1.0,0.0) == 0.0
    assert activations.multiparam_tanh_log1p_activation(0.0,1.0,-0.5) == 0.0
    assert activations.multiparam_tanh_log1p_activation(0.0,1.0,-1.0) == 0.0
    assert activations.multiparam_tanh_log1p_activation(-1.0,0.75,1.0) == -1.0
    assert activations.multiparam_tanh_log1p_activation(0.0,0.75,1.0) == 0.0
    assert activations.multiparam_tanh_log1p_activation(1.0,0.75,1.0) == 1.0
    assert activations.multiparam_tanh_log1p_activation(0.0,0.75,0.0) == 0.0
    assert activations.multiparam_tanh_log1p_activation(0.0,0.75,-1.0) == 0.0
    assert activations.multiparam_tanh_log1p_activation(-1.0,0.5,1.0) == -1.0
    assert activations.multiparam_tanh_log1p_activation(-0.5,0.5,1.0) == -0.5
    assert activations.multiparam_tanh_log1p_activation(0.0,0.5,1.0) == 0.0
    assert activations.multiparam_tanh_log1p_activation(0.5,0.5,1.0) == 0.5
    assert activations.multiparam_tanh_log1p_activation(1.0,0.5,1.0) == 1.0
    assert activations.multiparam_tanh_log1p_activation(0.0,0.5,0.5) == 0.0
    assert activations.multiparam_tanh_log1p_activation(0.0,0.5,0.0) == 0.0
    assert activations.multiparam_tanh_log1p_activation(0.0,0.5,-0.5) == 0.0
    assert activations.multiparam_tanh_log1p_activation(0.0,0.5,-1.0) == 0.0
    assert activations.multiparam_tanh_log1p_activation(-1.0,0.25,1.0) == -1.0
    assert activations.multiparam_tanh_log1p_activation(0.0,0.25,1.0) == 0.0
    assert activations.multiparam_tanh_log1p_activation(1.0,0.25,1.0) == 1.0
    assert activations.multiparam_tanh_log1p_activation(0.0,0.25,0.0) == 0.0
    assert activations.multiparam_tanh_log1p_activation(0.0,0.25,-1.0) == 0.0
    assert activations.multiparam_tanh_log1p_activation(-1.0,0.0,1.0) == -1.0
    assert activations.multiparam_tanh_log1p_activation(-0.5,0.0,1.0) == -0.5
    assert activations.multiparam_tanh_log1p_activation(0.0,0.0,1.0) == 0.0
    assert activations.multiparam_tanh_log1p_activation(0.5,0.0,1.0) == 0.5
    assert activations.multiparam_tanh_log1p_activation(1.0,0.0,1.0) == 1.0
    assert activations.multiparam_tanh_log1p_activation(0.0,0.0,0.5) == 0.0
    assert activations.multiparam_tanh_log1p_activation(0.0,0.0,0.0) == 0.0
    assert activations.multiparam_tanh_log1p_activation(0.0,0.0,-0.5) == 0.0
    assert activations.multiparam_tanh_log1p_activation(-1.0,0.0,-1.0) == -1.0
    assert activations.multiparam_tanh_log1p_activation(-0.5,0.0,-1.0) == -1.0
    assert activations.multiparam_tanh_log1p_activation(0.0,0.0,-1.0) == 0.0
    assert activations.multiparam_tanh_log1p_activation(0.5,0.0,-1.0) == 1.0
    assert activations.multiparam_tanh_log1p_activation(1.0,0.0,-1.0) == 1.0
    assert_almost_equal(activations.multiparam_tanh_log1p_activation(-0.5,1.0,-1.0),
                        -1*activations.multiparam_tanh_log1p_activation(0.5,1.0,-1.0))
    assert_almost_equal(activations.multiparam_tanh_log1p_activation(-1.0,1.0,-1.0),
                        -1*activations.multiparam_tanh_log1p_activation(1.0,1.0,-1.0))
    assert_almost_equal(activations.multiparam_tanh_log1p_activation(-0.5,1.0,0.0),
                        -1*activations.multiparam_tanh_log1p_activation(0.5,1.0,0.0))
    assert_almost_equal(activations.multiparam_tanh_log1p_activation(-1.0,0.75,-1.0),
                        -1*activations.multiparam_tanh_log1p_activation(1.0,0.75,-1.0))
    assert_almost_equal(activations.multiparam_tanh_log1p_activation(-0.5,0.5,-1.0),
                        -1*activations.multiparam_tanh_log1p_activation(0.5,0.5,-1.0))
    assert_almost_equal(activations.multiparam_tanh_log1p_activation(-1.0,1.0,-0.5),
                        -1*activations.multiparam_tanh_log1p_activation(1.0,1.0,-0.5))
    assert_almost_equal(activations.multiparam_tanh_log1p_activation(-0.5,0.5,0.0),
                        -1*activations.multiparam_tanh_log1p_activation(0.5,0.5,0.0))
    assert_almost_equal(activations.multiparam_tanh_log1p_activation(-1.0,0.5,-1.0),
                        -1*activations.multiparam_tanh_log1p_activation(1.0,0.5,-1.0))
    assert_almost_equal(activations.multiparam_tanh_log1p_activation(-0.5,0.0,0.0),
                        -1*activations.multiparam_tanh_log1p_activation(0.5,0.0,0.0))
    assert_almost_equal(activations.multiparam_tanh_log1p_activation(-1.0,0.5,-0.5),
                        -1*activations.multiparam_tanh_log1p_activation(1.0,0.5,-0.5))
    assert_almost_equal(activations.multiparam_tanh_log1p_activation(-1.0,0.25,-1.0),
                        -1*activations.multiparam_tanh_log1p_activation(1.0,0.25,-1.0))
    assert_almost_equal(activations.multiparam_tanh_log1p_activation(-1.0,1.0,0.0),
                        -1*activations.multiparam_tanh_log1p_activation(1.0,1.0,0.0))
    assert_almost_equal(activations.multiparam_tanh_log1p_activation(-1.0,0.75,0.0),
                        -1*activations.multiparam_tanh_log1p_activation(1.0,0.75,0.0))
    assert_almost_equal(activations.multiparam_tanh_log1p_activation(-1.0,0.5,0.0),
                        -1*activations.multiparam_tanh_log1p_activation(1.0,0.5,0.0))
    assert_almost_equal(activations.multiparam_tanh_log1p_activation(-1.0,0.25,0.0),
                        -1*activations.multiparam_tanh_log1p_activation(1.0,0.25,0.0))
    assert_almost_equal(activations.multiparam_tanh_log1p_activation(-1.0,0.0,0.0),
                        -1*activations.multiparam_tanh_log1p_activation(1.0,0.0,0.0))
    assert_almost_equal(activations.multiparam_tanh_log1p_activation(-1.0,1.0,0.5),
                        -1*activations.multiparam_tanh_log1p_activation(1.0,1.0,0.5))
    assert_almost_equal(activations.multiparam_tanh_log1p_activation(-1.0,0.5,0.5),
                        -1*activations.multiparam_tanh_log1p_activation(1.0,0.5,0.5))
    assert_almost_equal(activations.multiparam_tanh_log1p_activation(-1.0,0.0,0.5),
                        activations.multiparam_tanh_log1p_activation(-1.0,0.0,-0.5))
    assert_almost_equal(activations.multiparam_tanh_log1p_activation(1.0,0.0,0.5),
                        activations.multiparam_tanh_log1p_activation(1.0,0.0,-0.5))

def test_clamped_log1p_step():
    assert activations.clamped_log1p_step_activation(0.0,curve=1.0,tilt=1.0) == 0.0
    assert activations.clamped_log1p_step_activation(0.0,curve=1.0,tilt=0.5) == 0.0
    assert activations.clamped_log1p_step_activation(0.0,curve=1.0,tilt=0.0) == 0.0
    assert activations.clamped_log1p_step_activation(0.0,curve=1.0,tilt=-0.5) == 0.0
    assert activations.clamped_log1p_step_activation(0.0,curve=1.0,tilt=-1.0) == 0.0
    assert activations.clamped_log1p_step_activation(0.0,curve=0.75,tilt=1.0) == 0.0
    assert activations.clamped_log1p_step_activation(0.0,curve=0.75,tilt=0.0) == 0.0
    assert activations.clamped_log1p_step_activation(0.0,curve=0.75,tilt=-1.0) == 0.0
    assert activations.clamped_log1p_step_activation(0.0,curve=0.5,tilt=1.0) == 0.0
    assert activations.clamped_log1p_step_activation(0.0,curve=0.5,tilt=0.5) == 0.0
    assert activations.clamped_log1p_step_activation(0.0,curve=0.5,tilt=0.0) == 0.0
    assert activations.clamped_log1p_step_activation(0.0,curve=0.5,tilt=-0.5) == 0.0
    assert activations.clamped_log1p_step_activation(0.0,curve=0.5,tilt=-1.0) == 0.0
    assert activations.clamped_log1p_step_activation(0.0,curve=0.25,tilt=1.0) == 0.0
    assert activations.clamped_log1p_step_activation(0.0,curve=0.25,tilt=0.0) == 0.0
    assert activations.clamped_log1p_step_activation(0.0,curve=0.25,tilt=-1.0) == 0.0
    assert activations.clamped_log1p_step_activation(0.0,curve=0.0,tilt=1.0) == 0.0
    assert activations.clamped_log1p_step_activation(0.0,curve=0.0,tilt=0.5) == 0.0
    assert activations.clamped_log1p_step_activation(-1.0,curve=0.0,tilt=0.0) == -1.0
    assert activations.clamped_log1p_step_activation(0.0,curve=0.0,tilt=0.0) == 0.0
    assert activations.clamped_log1p_step_activation(1.0,curve=0.0,tilt=0.0) == 1.0
    assert activations.clamped_log1p_step_activation(-1.0,curve=0.0,tilt=-0.5) == -1.0
    assert activations.clamped_log1p_step_activation(0.0,curve=0.0,tilt=-0.5) == 0.0
    assert activations.clamped_log1p_step_activation(1.0,curve=0.0,tilt=-0.5) == 1.0
    assert activations.clamped_log1p_step_activation(-1.0,curve=0.0,tilt=-1.0) == -1.0
    assert activations.clamped_log1p_step_activation(-0.5,curve=0.0,tilt=-1.0) == -1.0
    assert activations.clamped_log1p_step_activation(0.0,curve=0.0,tilt=-1.0) == 0.0
    assert activations.clamped_log1p_step_activation(0.5,curve=0.0,tilt=-1.0) == 1.0
    assert activations.clamped_log1p_step_activation(1.0,curve=0.0,tilt=-1.0) == 1.0
    assert_almost_equal(activations.clamped_log1p_step_activation(-0.5,curve=0.0,tilt=1.0),
                        -1*activations.clamped_log1p_step_activation(0.5,curve=0.0,tilt=1.0))
    assert_almost_equal(activations.clamped_log1p_step_activation(-0.5,curve=1.0,tilt=-1.0),
                        -1*activations.clamped_log1p_step_activation(0.5,curve=1.0,tilt=-1.0))
    assert_almost_equal(activations.clamped_log1p_step_activation(-1.0,curve=1.0,tilt=-1.0),
                        -1*activations.clamped_log1p_step_activation(1.0,curve=1.0,tilt=-1.0))
    assert_almost_equal(activations.clamped_log1p_step_activation(-0.5,curve=0.5,tilt=1.0),
                        -1*activations.clamped_log1p_step_activation(0.5,curve=0.5,tilt=1.0))
    assert_almost_equal(activations.clamped_log1p_step_activation(-0.5,curve=1.0,tilt=0.0),
                        -1*activations.clamped_log1p_step_activation(0.5,curve=1.0,tilt=0.0))
    assert_almost_equal(activations.clamped_log1p_step_activation(-1.0,curve=0.75,tilt=-1.0),
                        -1*activations.clamped_log1p_step_activation(1.0,curve=0.75,tilt=-1.0))
    assert_almost_equal(activations.clamped_log1p_step_activation(-0.5,curve=0.5,tilt=0.0),
                        -1*activations.clamped_log1p_step_activation(0.5,curve=0.5,tilt=0.0))
    assert_almost_equal(activations.clamped_log1p_step_activation(-0.5,curve=0.5,tilt=-1.0),
                        -1*activations.clamped_log1p_step_activation(0.5,curve=0.5,tilt=-1.0))
    assert_almost_equal(activations.clamped_log1p_step_activation(-1.0,curve=1.0,tilt=-0.5),
                        -1*activations.clamped_log1p_step_activation(1.0,curve=1.0,tilt=-0.5))
    assert_almost_equal(activations.clamped_log1p_step_activation(-1.0,curve=0.5,tilt=-1.0),
                        -1*activations.clamped_log1p_step_activation(1.0,curve=0.5,tilt=-1.0))
    assert_almost_equal(activations.clamped_log1p_step_activation(-0.5,curve=1.0,tilt=1.0),
                        -1*activations.clamped_log1p_step_activation(0.5,curve=1.0,tilt=1.0))
    assert_almost_equal(activations.clamped_log1p_step_activation(-1.0,curve=0.25,tilt=1.0),
                        -1*activations.clamped_log1p_step_activation(1.0,curve=0.25,tilt=1.0))
    assert_almost_equal(activations.clamped_log1p_step_activation(-1.0,curve=0.5,tilt=-0.5),
                        -1*activations.clamped_log1p_step_activation(1.0,curve=0.5,tilt=-0.5))
    assert_almost_equal(activations.clamped_log1p_step_activation(-1.0,curve=0.25,tilt=-1.0),
                        -1*activations.clamped_log1p_step_activation(1.0,curve=0.25,tilt=-1.0))
    assert_almost_equal(activations.clamped_log1p_step_activation(-1.0,curve=0.0,tilt=0.5),
                        -1*activations.clamped_log1p_step_activation(1.0,curve=0.0,tilt=0.5))
    assert_almost_equal(activations.clamped_log1p_step_activation(-1.0,curve=1.0,tilt=0.0),
                        -1*activations.clamped_log1p_step_activation(1.0,curve=1.0,tilt=0.0))
    assert_almost_equal(activations.clamped_log1p_step_activation(-1.0,curve=0.75,tilt=0.0),
                        -1*activations.clamped_log1p_step_activation(1.0,curve=0.75,tilt=0.0))
    assert_almost_equal(activations.clamped_log1p_step_activation(-1.0,curve=0.5,tilt=0.0),
                        -1*activations.clamped_log1p_step_activation(1.0,curve=0.5,tilt=0.0))
    assert_almost_equal(activations.clamped_log1p_step_activation(-1.0,curve=0.25,tilt=0.0),
                        -1*activations.clamped_log1p_step_activation(1.0,curve=0.25,tilt=0.0))
    assert_almost_equal(activations.clamped_log1p_step_activation(-1.0,curve=0.5,tilt=1.0),
                        -1*activations.clamped_log1p_step_activation(1.0,curve=0.5,tilt=1.0))
    assert_almost_equal(activations.clamped_log1p_step_activation(-1.0,curve=0.5,tilt=0.5),
                        -1*activations.clamped_log1p_step_activation(1.0,curve=0.5,tilt=0.5))
    assert_almost_equal(activations.clamped_log1p_step_activation(-1.0,curve=1.0,tilt=0.5),
                        -1*activations.clamped_log1p_step_activation(1.0,curve=1.0,tilt=0.5))
    assert_almost_equal(activations.clamped_log1p_step_activation(-1.0,curve=0.75,tilt=1.0),
                        -1*activations.clamped_log1p_step_activation(1.0,curve=0.75,tilt=1.0))
    assert_almost_equal(activations.clamped_log1p_step_activation(-1.0,curve=1.0,tilt=1.0),
                        -1*activations.clamped_log1p_step_activation(1.0,curve=1.0,tilt=1.0))
    assert_almost_equal(activations.clamped_log1p_step_activation(-1.0,curve=0.0,tilt=1.0),
                        activations.clamped_log1p_step_activation(-0.5,curve=0.0,tilt=0.0))
    assert_almost_equal(activations.clamped_log1p_step_activation(1.0,curve=0.0,tilt=1.0),
                        activations.clamped_log1p_step_activation(0.5,curve=0.0,tilt=0.0))


def test_multiparam_pow():
    assert activations.multiparam_pow_activation(-1.0,1.0) == -1.0
    assert activations.multiparam_pow_activation(-0.5,1.0) == -0.0625
    assert activations.multiparam_pow_activation(0.0,1.0) == 0.0
    assert activations.multiparam_pow_activation(0.5,1.0) == 0.0625
    assert activations.multiparam_pow_activation(1.0,1.0) == 1.0
    assert activations.multiparam_pow_activation(-1.0,0.5) == -1.0
    assert activations.multiparam_pow_activation(0.0,0.5) == 0.0
    assert activations.multiparam_pow_activation(1.0,0.5) == 1.0
    assert activations.multiparam_pow_activation(-1.0,0.0) == -1.0
    assert activations.multiparam_pow_activation(-0.75,0.0) == -0.75
    assert activations.multiparam_pow_activation(-0.5,0.0) == -0.5
    assert activations.multiparam_pow_activation(-0.25,0.0) == -0.25
    assert activations.multiparam_pow_activation(0.0,0.0) == 0.0
    assert activations.multiparam_pow_activation(0.25,0.0) == 0.25
    assert activations.multiparam_pow_activation(0.5,0.0) == 0.5
    assert activations.multiparam_pow_activation(0.75,0.0) == 0.75
    assert activations.multiparam_pow_activation(1.0,0.0) == 1.0
    assert activations.multiparam_pow_activation(-1.0,-0.5) == -1.0
    assert activations.multiparam_pow_activation(0.0,-0.5) == 0.0
    assert activations.multiparam_pow_activation(1.0,-0.5) == 1.0
    assert activations.multiparam_pow_activation(-1.0,-1.0) == -1.0
    assert activations.multiparam_pow_activation(0.0,-1.0) == 0.0
    assert activations.multiparam_pow_activation(1.0,-1.0) == 1.0
    assert_almost_equal(activations.multiparam_pow_activation(-0.5,0.5),
                        -1*activations.multiparam_pow_activation(0.5,0.5))
    assert_almost_equal(activations.multiparam_pow_activation(-0.5,-0.5),
                        -1*activations.multiparam_pow_activation(0.5,-0.5))
    assert_almost_equal(activations.multiparam_pow_activation(-0.5,-1.0),
                        -1*activations.multiparam_pow_activation(0.5,-1.0))

def test_wave():
    assert activations.wave_activation(-1.0,1.0) == 1.0
    assert activations.wave_activation(-0.5,1.0) == -1.0
    assert activations.wave_activation(0.0,1.0) == 0.0
    assert activations.wave_activation(0.5,1.0) == 1.0
    assert activations.wave_activation(1.0,1.0) == -1.0
    assert activations.wave_activation(0.0,0.75) == 0.0
    assert activations.wave_activation(0.0,0.5) == 0.0
    assert activations.wave_activation(0.0,0.25) == 0.0
    assert activations.wave_activation(0.0,0.0) == 0.0
    assert_almost_equal(activations.wave_activation(-0.5,0.0),
                        -1*activations.wave_activation(0.5,0.0))
    assert_almost_equal(activations.wave_activation(-0.5,0.25),
                        -1*activations.wave_activation(0.5,0.25))
    assert_almost_equal(activations.wave_activation(-0.75,0.5),
                        -1*activations.wave_activation(0.75,0.5))
    assert_almost_equal(activations.wave_activation(-0.5,0.5),
                        -1*activations.wave_activation(0.5,0.5))
    assert_almost_equal(activations.wave_activation(-0.5,0.75),
                        -1*activations.wave_activation(0.5,0.75))
    assert_almost_equal(activations.wave_activation(-1.0,0.0),
                        -1*activations.wave_activation(1.0,0.0))
    assert_almost_equal(activations.wave_activation(-1.0,0.25),
                        -1*activations.wave_activation(1.0,0.25))
    assert_almost_equal(activations.wave_activation(-0.25,0.5),
                        -1*activations.wave_activation(0.25,0.5))
    assert_almost_equal(activations.wave_activation(-1.0,0.5),
                        -1*activations.wave_activation(1.0,0.5))
    assert_almost_equal(activations.wave_activation(-1.0,0.75),
                        -1*activations.wave_activation(1.0,0.75))

def test_multiparam_tanh_approx():
    assert activations.multiparam_tanh_approx_activation(0.0,1.26,1.0) == 0.0
    assert activations.multiparam_tanh_approx_activation(0.0,1.26,0.0) == 0.0
    assert activations.multiparam_tanh_approx_activation(0.0,1.26,-1.0) == 0.0
    assert activations.multiparam_tanh_approx_activation(0.0,0.51,1.0) == 0.0
    assert activations.multiparam_tanh_approx_activation(0.0,0.51,0.5) == 0.0
    assert activations.multiparam_tanh_approx_activation(0.0,0.51,0.0) == 0.0
    assert activations.multiparam_tanh_approx_activation(0.0,0.51,-0.5) == 0.0
    assert activations.multiparam_tanh_approx_activation(0.0,0.51,-1.0) == 0.0
    assert activations.multiparam_tanh_approx_activation(0.0,-0.24,1.0) == 0.0
    assert activations.multiparam_tanh_approx_activation(0.0,-0.24,0.5) == 0.0
    assert activations.multiparam_tanh_approx_activation(0.0,-0.24,0.0) == 0.0
    assert activations.multiparam_tanh_approx_activation(-1.0,-0.24,-0.5) == -1.0
    assert activations.multiparam_tanh_approx_activation(0.0,-0.24,-0.5) == 0.0
    assert activations.multiparam_tanh_approx_activation(1.0,-0.24,-0.5) == 1.0
    assert activations.multiparam_tanh_approx_activation(-1.0,-0.24,-1.0) == -1.0
    assert activations.multiparam_tanh_approx_activation(-0.5,-0.24,-1.0) == -1.0
    assert activations.multiparam_tanh_approx_activation(0.0,-0.24,-1.0) == 0.0
    assert activations.multiparam_tanh_approx_activation(0.5,-0.24,-1.0) == 1.0
    assert activations.multiparam_tanh_approx_activation(1.0,-0.24,-1.0) == 1.0
    assert activations.multiparam_tanh_approx_activation(0.0,-0.99,1.0) == 0.0
    assert activations.multiparam_tanh_approx_activation(0.0,-0.99,0.5) == 0.0
    assert activations.multiparam_tanh_approx_activation(0.0,-0.99,0.0) == 0.0
    assert activations.multiparam_tanh_approx_activation(-1.0,-0.99,-0.5) == -1.0
    assert activations.multiparam_tanh_approx_activation(0.0,-0.99,-0.5) == 0.0
    assert activations.multiparam_tanh_approx_activation(1.0,-0.99,-0.5) == 1.0
    assert activations.multiparam_tanh_approx_activation(-1.0,-0.99,-1.0) == -1.0
    assert activations.multiparam_tanh_approx_activation(-0.5,-0.99,-1.0) == -1.0
    assert activations.multiparam_tanh_approx_activation(0.0,-0.99,-1.0) == 0.0
    assert activations.multiparam_tanh_approx_activation(0.5,-0.99,-1.0) == 1.0
    assert activations.multiparam_tanh_approx_activation(1.0,-0.99,-1.0) == 1.0
    assert activations.multiparam_tanh_approx_activation(0.0,-1.74,1.0) == 0.0
    assert activations.multiparam_tanh_approx_activation(-1.0,-1.74,0.0) == -1.0
    assert activations.multiparam_tanh_approx_activation(0.0,-1.74,0.0) == 0.0
    assert activations.multiparam_tanh_approx_activation(1.0,-1.74,0.0) == 1.0
    assert activations.multiparam_tanh_approx_activation(-1.0,-1.74,-1.0) == -1.0
    assert activations.multiparam_tanh_approx_activation(0.0,-1.74,-1.0) == 0.0
    assert activations.multiparam_tanh_approx_activation(1.0,-1.74,-1.0) == 1.0
    assert_almost_equal(activations.multiparam_tanh_approx_activation(-1.0,1.26,0.0),
                        -1*activations.multiparam_tanh_approx_activation(1.0,1.26,0.0))
    assert_almost_equal(activations.multiparam_tanh_approx_activation(-0.5,0.51,0.0),
                        -1*activations.multiparam_tanh_approx_activation(0.5,0.51,0.0))
    assert_almost_equal(activations.multiparam_tanh_approx_activation(-1.0,1.26,1.0),
                        -1*activations.multiparam_tanh_approx_activation(1.0,1.26,1.0))
    assert_almost_equal(activations.multiparam_tanh_approx_activation(-1.0,0.51,0.0),
                        -1*activations.multiparam_tanh_approx_activation(1.0,0.51,0.0))
    assert_almost_equal(activations.multiparam_tanh_approx_activation(-0.5,-0.24,0.0),
                        -1*activations.multiparam_tanh_approx_activation(0.5,-0.24,0.0))
    assert_almost_equal(activations.multiparam_tanh_approx_activation(-1.0,-0.24,0.0),
                        -1*activations.multiparam_tanh_approx_activation(1.0,-0.24,0.0))
    assert_almost_equal(activations.multiparam_tanh_approx_activation(-1.0,1.26,-1.0),
                        -1*activations.multiparam_tanh_approx_activation(1.0,1.26,-1.0))
    assert_almost_equal(activations.multiparam_tanh_approx_activation(-1.0,0.51,-0.5),
                        -1*activations.multiparam_tanh_approx_activation(1.0,0.51,-0.5))
    assert_almost_equal(activations.multiparam_tanh_approx_activation(-0.5,-0.99,0.0),
                        -1*activations.multiparam_tanh_approx_activation(0.5,-0.99,0.0))
    assert_almost_equal(activations.multiparam_tanh_approx_activation(-0.5,0.51,-1.0),
                        -1*activations.multiparam_tanh_approx_activation(0.5,0.51,-1.0))
    assert_almost_equal(activations.multiparam_tanh_approx_activation(-1.0,-0.99,0.0),
                        -1*activations.multiparam_tanh_approx_activation(1.0,-0.99,0.0))
    assert_almost_equal(activations.multiparam_tanh_approx_activation(-1.0,0.51,-1.0),
                        -1*activations.multiparam_tanh_approx_activation(1.0,0.51,-1.0))
    assert_almost_equal(activations.multiparam_tanh_approx_activation(-1.0,0.51,0.5),
                        -1*activations.multiparam_tanh_approx_activation(1.0,0.51,0.5))
    assert_almost_equal(activations.multiparam_tanh_approx_activation(-0.5,0.51,1.0),
                        -1*activations.multiparam_tanh_approx_activation(0.5,0.51,1.0))
    assert_almost_equal(activations.multiparam_tanh_approx_activation(-1.0,-0.24,0.5),
                        -1*activations.multiparam_tanh_approx_activation(1.0,-0.24,0.5))
    assert_almost_equal(activations.multiparam_tanh_approx_activation(-1.0,-0.99,0.5),
                        -1*activations.multiparam_tanh_approx_activation(1.0,-0.99,0.5))
    assert_almost_equal(activations.multiparam_tanh_approx_activation(-1.0,0.51,1.0),
                        -1*activations.multiparam_tanh_approx_activation(1.0,0.51,1.0))
    assert_almost_equal(activations.multiparam_tanh_approx_activation(-0.5,-0.24,1.0),
                        -1*activations.multiparam_tanh_approx_activation(0.5,-0.24,1.0))
    assert_almost_equal(activations.multiparam_tanh_approx_activation(-1.0,-0.24,1.0),
                        -1*activations.multiparam_tanh_approx_activation(1.0,-0.24,1.0))



def test_multiparam_sigmoid_approx():
    assert activations.multiparam_sigmoid_approx_activation(0.0,1.26) == 0.5
    assert activations.multiparam_sigmoid_approx_activation(0.0,0.51) == 0.5
    assert activations.multiparam_sigmoid_approx_activation(0.0,-0.24) == 0.5
    assert activations.multiparam_sigmoid_approx_activation(0.0,-0.99) == 0.5
    assert activations.multiparam_sigmoid_approx_activation(-1.0,-1.74) == 0.0
    assert activations.multiparam_sigmoid_approx_activation(-0.5,-1.74) == 0.0
    assert activations.multiparam_sigmoid_approx_activation(0.0,-1.74) == 0.5
    assert activations.multiparam_sigmoid_approx_activation(0.5,-1.74) == 1.0
    assert activations.multiparam_sigmoid_approx_activation(1.0,-1.74) == 1.0
    assert_almost_equal(activations.multiparam_sigmoid_approx_activation(-1.0,-0.99),(17/8005))
    assert_almost_equal(activations.multiparam_sigmoid_approx_activation(-1.0,-0.24),(268/7543))
    assert_almost_equal(activations.multiparam_sigmoid_approx_activation(-0.5,-0.24),(277/3725))
    assert_almost_equal(activations.multiparam_sigmoid_approx_activation(-1.0,0.51),(1203/9290))
    assert_almost_equal(activations.multiparam_sigmoid_approx_activation(-0.5,0.51),(1302/6217))
    assert_almost_equal(activations.multiparam_sigmoid_approx_activation(-1.0,1.26),(900/3103))
    assert_almost_equal(activations.multiparam_sigmoid_approx_activation(1.0,1.26),(2203/3103))
    assert_almost_equal(activations.multiparam_sigmoid_approx_activation(0.5,0.51),(4915/6217))
    assert_almost_equal(activations.multiparam_sigmoid_approx_activation(1.0,0.51),(8087/9290))
    assert_almost_equal(activations.multiparam_sigmoid_approx_activation(0.5,-0.24),(3448/3725))
    assert_almost_equal(activations.multiparam_sigmoid_approx_activation(1.0,-0.24),(7275/7543))
    assert_almost_equal(activations.multiparam_sigmoid_approx_activation(1.0,-0.99),(7988/8005))


def test_multiparam_gauss():
    assert 0.0 <= activations.multiparam_gauss_activation(-1.0,1.0,1.0) <= 0.000123
    assert 0.0 <= activations.multiparam_gauss_activation(1.0,1.0,1.0) <= 0.000123
    assert 0.0 <= activations.multiparam_gauss_activation(-1.0,0.75,1.0) <= 0.000123
    assert 0.0 <= activations.multiparam_gauss_activation(1.0,0.75,1.0) <= 0.000123
    assert 0.0 <= activations.multiparam_gauss_activation(-1.0,0.5,1.0) <= 0.000123
    assert 0.0 <= activations.multiparam_gauss_activation(1.0,0.5,1.0) <= 0.000123
    assert 0.0 <= activations.multiparam_gauss_activation(-1.0,0.25,1.0) <= 0.000123
    assert 0.0 <= activations.multiparam_gauss_activation(1.0,0.25,1.0) <= 0.000123
    assert 0.0 <= activations.multiparam_gauss_activation(-1.0,0.0,1.0) <= 0.000123
    assert 0.0 <= activations.multiparam_gauss_activation(1.0,0.0,1.0) <= 0.000123

    for a in [0.0, 0.25, 0.5, 0.75, 1.0]:
        for b in [0.0, 0.25, 0.5, 0.75, 1.0]:
            assert activations.multiparam_gauss_activation(0.0,a,b) == 1.0
            for z in [0.25,0.5,0.75,1.0]:
                assert_almost_equal(activations.multiparam_gauss_activation(z,a,b),
                                    activations.multiparam_gauss_activation(-z,a,b))

def test_bicentral():
    assert activations.bicentral_activation(-1.0,lower=0.5,tilt=1.0,width=0.5) == 0.0
    assert activations.bicentral_activation(0.5,lower=0.5,tilt=1.0,width=0.5) == 0.5
    assert activations.bicentral_activation(-0.5,lower=0.5,tilt=-1.0,width=0.5) == 0.5
    assert activations.bicentral_activation(1.0,lower=0.5,tilt=-1.0,width=0.5) == 0.0
    assert activations.bicentral_activation(-1.0,lower=0.25,tilt=1.0,width=0.5) == 0.0
    assert activations.bicentral_activation(1.0,lower=0.25,tilt=-1.0,width=0.5) == 0.0
    assert activations.bicentral_activation(-1.0,lower=0.0,tilt=1.0,width=0.5) == 0.0
    assert activations.bicentral_activation(0.5,lower=0.0,tilt=1.0,width=0.5) == 0.5
    assert activations.bicentral_activation(-1.0,lower=0.0,tilt=0.5,width=0.5) == 0.0
    assert activations.bicentral_activation(-1.0,lower=0.0,tilt=0.0,width=0.5) == 0.0
    assert activations.bicentral_activation(-0.5,lower=0.0,tilt=0.0,width=0.5) == 0.5
    assert activations.bicentral_activation(0.0,lower=0.0,tilt=0.0,width=0.5) == 1.0
    assert activations.bicentral_activation(0.5,lower=0.0,tilt=0.0,width=0.5) == 0.5
    assert activations.bicentral_activation(1.0,lower=0.0,tilt=0.0,width=0.5) == 0.0
    assert activations.bicentral_activation(1.0,lower=0.0,tilt=-0.5,width=0.5) == 0.0
    assert activations.bicentral_activation(-0.5,lower=0.0,tilt=-1.0,width=0.5) == 0.5
    assert activations.bicentral_activation(1.0,lower=0.0,tilt=-1.0,width=0.5) == 0.0
    assert activations.bicentral_activation(0.0,lower=1.0,tilt=0.0,width=0.0) == 0.125
    assert activations.bicentral_activation(0.0,lower=0.75,tilt=0.0,width=0.0) == 0.1875
    assert activations.bicentral_activation(0.0,lower=0.5,tilt=0.0,width=0.0) == 0.25
    assert activations.bicentral_activation(-1.0,lower=0.25,tilt=0.0,width=1.0) == 0.5
    assert activations.bicentral_activation(1.0,lower=0.25,tilt=0.0,width=1.0) == 0.5
    assert 0.0 <= activations.bicentral_activation(-1.0,lower=0.25,tilt=0.0,width=0.0) <= 0.000123
    assert activations.bicentral_activation(0.0,lower=0.25,tilt=0.0,width=0.0) == 0.25
    assert 0.0 <= activations.bicentral_activation(1.0,lower=0.25,tilt=0.0,width=0.0) <= 0.000123
    assert activations.bicentral_activation(-1.0,lower=0.0,tilt=0.0,width=1.0) == 0.5
    assert activations.bicentral_activation(-0.5,lower=0.0,tilt=0.0,width=1.0) == 1.0
    assert activations.bicentral_activation(0.0,lower=0.0,tilt=0.0,width=1.0) == 1.0
    assert activations.bicentral_activation(0.5,lower=0.0,tilt=0.0,width=1.0) == 1.0
    assert activations.bicentral_activation(1.0,lower=0.0,tilt=0.0,width=1.0) == 0.5
    assert activations.bicentral_activation(-1.0,lower=0.0,tilt=0.0,width=0.75) == 0.0
    assert activations.bicentral_activation(0.0,lower=0.0,tilt=0.0,width=0.75) == 1.0
    assert activations.bicentral_activation(1.0,lower=0.0,tilt=0.0,width=0.75) == 0.0
    assert activations.bicentral_activation(-1.0,lower=0.0,tilt=0.0,width=0.25) == 0.0
    assert activations.bicentral_activation(0.0,lower=0.0,tilt=0.0,width=0.25) == 1.0
    assert activations.bicentral_activation(1.0,lower=0.0,tilt=0.0,width=0.25) == 0.0
    assert activations.bicentral_activation(-1.0,lower=0.0,tilt=0.0,width=0.0) == 0.0
    assert activations.bicentral_activation(-0.5,lower=0.0,tilt=0.0,width=0.0) == 0.0
    assert activations.bicentral_activation(0.0,lower=0.0,tilt=0.0,width=0.0) == 0.25
    assert activations.bicentral_activation(0.5,lower=0.0,tilt=0.0,width=0.0) == 0.0
    assert activations.bicentral_activation(1.0,lower=0.0,tilt=0.0,width=0.0) == 0.0
    assert activations.bicentral_activation(1.0,lower=0.5,tilt=1.0,width=1.0) == 0.5
    assert activations.bicentral_activation(-1.0,lower=0.5,tilt=1.0,width=0.75) == 0.0
    assert activations.bicentral_activation(-1.0,lower=0.5,tilt=1.0,width=0.25) == 0.0
    assert activations.bicentral_activation(-1.0,lower=0.5,tilt=1.0,width=0.0) == 0.0
    assert activations.bicentral_activation(-0.5,lower=0.5,tilt=1.0,width=0.0) == 0.0
    assert activations.bicentral_activation(0.0,lower=0.5,tilt=1.0,width=0.0) == 0.25
    assert activations.bicentral_activation(1.0,lower=0.5,tilt=0.5,width=1.0) == 0.5
    assert 0.0 <= activations.bicentral_activation(-1.0,lower=0.5,tilt=0.5,width=0.0) <= 0.000123
    assert activations.bicentral_activation(0.0,lower=0.5,tilt=0.5,width=0.0) == 0.25
    assert activations.bicentral_activation(-1.0,lower=0.5,tilt=-0.5,width=1.0) == 0.5
    assert activations.bicentral_activation(0.0,lower=0.5,tilt=-0.5,width=0.0) == 0.25
    assert 0.0 <= activations.bicentral_activation(1.0,lower=0.5,tilt=-0.5,width=0.0) <= 0.000123
    assert activations.bicentral_activation(-1.0,lower=0.5,tilt=-1.0,width=1.0) == 0.5
    assert activations.bicentral_activation(1.0,lower=0.5,tilt=-1.0,width=0.75) == 0.0
    assert activations.bicentral_activation(1.0,lower=0.5,tilt=-1.0,width=0.25) == 0.0
    assert activations.bicentral_activation(0.0,lower=0.5,tilt=-1.0,width=0.0) == 0.25
    assert activations.bicentral_activation(0.5,lower=0.5,tilt=-1.0,width=0.0) == 0.0
    assert activations.bicentral_activation(1.0,lower=0.5,tilt=-1.0,width=0.0) == 0.0
    assert_almost_equal(activations.bicentral_activation(-1.0,lower=0.75,tilt=1.0,width=0.5),
                        activations.bicentral_activation(1.0,lower=0.75,tilt=-1.0,width=0.5))
    assert_almost_equal(activations.bicentral_activation(-1.0,lower=0.5,tilt=0.5,width=0.5),
                        activations.bicentral_activation(1.0,lower=0.5,tilt=-0.5,width=0.5))
    assert_almost_equal(activations.bicentral_activation(-1.0,lower=0.5,tilt=0.0,width=0.0),
                        activations.bicentral_activation(1.0,lower=0.5,tilt=0.0,width=0.0))
    assert_almost_equal(activations.bicentral_activation(-1.0,lower=1.0,tilt=1.0,width=0.5),
                        activations.bicentral_activation(1.0,lower=1.0,tilt=-1.0,width=0.5))
    assert_almost_equal(activations.bicentral_activation(-1.0,lower=0.5,tilt=0.0,width=0.25),
                        activations.bicentral_activation(1.0,lower=0.5,tilt=0.0,width=0.25))
    assert_almost_equal(activations.bicentral_activation(-1.0,lower=1.0,tilt=0.5,width=0.5),
                        activations.bicentral_activation(1.0,lower=1.0,tilt=-0.5,width=0.5))
    assert_almost_equal(activations.bicentral_activation(-0.5,lower=0.5,tilt=0.0,width=0.0),
                        activations.bicentral_activation(0.5,lower=0.5,tilt=0.0,width=0.0))
    assert_almost_equal(activations.bicentral_activation(-1.0,lower=0.5,tilt=0.0,width=0.5),
                        activations.bicentral_activation(1.0,lower=0.5,tilt=0.0,width=0.5))
    assert_almost_equal(activations.bicentral_activation(1.0,lower=0.0,tilt=1.0,width=0.5),
                        activations.bicentral_activation(-1.0,lower=0.0,tilt=-1.0,width=0.5))
    assert_almost_equal(activations.bicentral_activation(-1.0,lower=0.75,tilt=0.0,width=0.0),
                        activations.bicentral_activation(1.0,lower=0.75,tilt=0.0,width=0.0))
    assert_almost_equal(activations.bicentral_activation(-1.0,lower=1.0,tilt=0.0,width=0.0),
                        activations.bicentral_activation(1.0,lower=1.0,tilt=0.0,width=0.0))
    assert_almost_equal(activations.bicentral_activation(-0.5,lower=1.0,tilt=0.0,width=0.0),
                        activations.bicentral_activation(0.5,lower=1.0,tilt=0.0,width=0.0))
    assert_almost_equal(activations.bicentral_activation(-1.0,lower=1.0,tilt=0.0,width=0.25),
                        activations.bicentral_activation(1.0,lower=1.0,tilt=0.0,width=0.25))
    assert_almost_equal(activations.bicentral_activation(-1.0,lower=0.5,tilt=0.0,width=0.75),
                        activations.bicentral_activation(1.0,lower=0.5,tilt=0.0,width=0.75))
    assert_almost_equal(activations.bicentral_activation(-1.0,lower=1.0,tilt=0.0,width=0.5),
                        activations.bicentral_activation(1.0,lower=1.0,tilt=0.0,width=0.5))
    assert_almost_equal(activations.bicentral_activation(1.0,lower=0.5,tilt=0.5,width=0.0),
                        activations.bicentral_activation(-1.0,lower=0.5,tilt=-0.5,width=0.0))
    assert_almost_equal(activations.bicentral_activation(-1.0,lower=0.75,tilt=0.0,width=0.5),
                        activations.bicentral_activation(1.0,lower=0.75,tilt=0.0,width=0.5))
    assert_almost_equal(activations.bicentral_activation(0.0,lower=1.0,tilt=0.0,width=0.25),(1067/6894))
    assert_almost_equal(activations.bicentral_activation(-1.0,lower=1.0,tilt=0.0,width=0.75),
                        activations.bicentral_activation(1.0,lower=1.0,tilt=0.0,width=0.75))
    assert_almost_equal(activations.bicentral_activation(-0.5,lower=1.0,tilt=0.0,width=0.5),
                        activations.bicentral_activation(0.5,lower=1.0,tilt=0.0,width=0.5))
    assert_almost_equal(activations.bicentral_activation(-1.0,lower=1.0,tilt=0.0,width=1.0),
                        activations.bicentral_activation(1.0,lower=1.0,tilt=0.0,width=1.0))
    assert_almost_equal(activations.bicentral_activation(0.0,lower=1.0,tilt=0.5,width=0.5),
                        activations.bicentral_activation(0.0,lower=1.0,tilt=-0.5,width=0.5))
    assert_almost_equal(activations.bicentral_activation(1.0,lower=1.0,tilt=0.5,width=0.5),
                        activations.bicentral_activation(-1.0,lower=1.0,tilt=-0.5,width=0.5))
    assert_almost_equal(activations.bicentral_activation(-0.5,lower=1.0,tilt=0.0,width=1.0),
                        activations.bicentral_activation(0.5,lower=1.0,tilt=0.0,width=1.0))
    assert_almost_equal(activations.bicentral_activation(-0.5,lower=1.0,tilt=1.0,width=0.5),
                        activations.bicentral_activation(0.5,lower=1.0,tilt=-1.0,width=0.5))
    assert_almost_equal(activations.bicentral_activation(1.0,lower=0.5,tilt=1.0,width=0.0),
                        activations.bicentral_activation(-1.0,lower=0.5,tilt=-1.0,width=0.0))
    assert_almost_equal(activations.bicentral_activation(-0.5,lower=0.5,tilt=1.0,width=0.5),
                        activations.bicentral_activation(0.5,lower=0.5,tilt=-1.0,width=0.5))
    assert_almost_equal(activations.bicentral_activation(-1.0,lower=0.75,tilt=0.0,width=1.0),
                        activations.bicentral_activation(1.0,lower=0.75,tilt=0.0,width=1.0))
    assert_almost_equal(activations.bicentral_activation(1.0,lower=0.5,tilt=1.0,width=0.25),
                        activations.bicentral_activation(-1.0,lower=0.5,tilt=-1.0,width=0.25))
    assert_almost_equal(activations.bicentral_activation(-1.0,lower=0.5,tilt=1.0,width=1.0),
                        activations.bicentral_activation(1.0,lower=0.5,tilt=-1.0,width=1.0))
    assert_almost_equal(activations.bicentral_activation(0.0,lower=0.75,tilt=0.0,width=0.5),(3396/7657))
    assert_almost_equal(activations.bicentral_activation(1.0,lower=0.5,tilt=1.0,width=0.75),
                        activations.bicentral_activation(-1.0,lower=0.5,tilt=-1.0,width=0.75))
    assert_almost_equal(activations.bicentral_activation(-1.0,lower=0.5,tilt=0.5,width=1.0),
                        activations.bicentral_activation(1.0,lower=0.5,tilt=-0.5,width=1.0))
    assert_almost_equal(activations.bicentral_activation(0.0,lower=1.0,tilt=1.0,width=0.5),
                        activations.bicentral_activation(0.0,lower=1.0,tilt=-1.0,width=0.5))
    assert_almost_equal(activations.bicentral_activation(1.0,lower=0.75,tilt=1.0,width=0.5),
                        activations.bicentral_activation(-1.0,lower=0.75,tilt=-1.0,width=0.5))
    assert_almost_equal(activations.bicentral_activation(1.0,lower=1.0,tilt=1.0,width=0.5),
                        activations.bicentral_activation(-1.0,lower=1.0,tilt=-1.0,width=0.5))
    assert_almost_equal(activations.bicentral_activation(-1.0,lower=0.5,tilt=0.0,width=1.0),
                        activations.bicentral_activation(1.0,lower=0.5,tilt=0.0,width=1.0))
    assert_almost_equal(activations.bicentral_activation(0.0,lower=0.75,tilt=1.0,width=0.5),
                        activations.bicentral_activation(0.0,lower=0.75,tilt=-1.0,width=0.5))
    assert_almost_equal(activations.bicentral_activation(0.0,lower=0.75,tilt=0.0,width=1.0),(1077/1976))
    assert_almost_equal(activations.bicentral_activation(0.0,lower=0.5,tilt=1.0,width=0.25),
                        activations.bicentral_activation(0.0,lower=0.5,tilt=-1.0,width=0.25))
    assert_almost_equal(activations.bicentral_activation(0.0,lower=0.5,tilt=1.0,width=0.75),
                        activations.bicentral_activation(0.0,lower=0.5,tilt=-1.0,width=0.75))
    assert_almost_equal(activations.bicentral_activation(0.0,lower=0.5,tilt=1.0,width=1.0),
                        activations.bicentral_activation(0.0,lower=0.5,tilt=-1.0,width=1.0))
    assert_almost_equal(activations.bicentral_activation(-0.5,lower=0.5,tilt=1.0,width=1.0),
                        activations.bicentral_activation(0.5,lower=0.5,tilt=-1.0,width=1.0))
    assert_almost_equal(activations.bicentral_activation(0.0,lower=0.5,tilt=0.5,width=0.5),
                        activations.bicentral_activation(0.0,lower=0.5,tilt=-0.5,width=0.5))
    assert_almost_equal(activations.bicentral_activation(0.0,lower=0.25,tilt=1.0,width=0.5),
                        activations.bicentral_activation(0.0,lower=0.25,tilt=-1.0,width=0.5))
    assert_almost_equal(activations.bicentral_activation(0.0,lower=0.5,tilt=0.5,width=1.0),
                        activations.bicentral_activation(0.0,lower=0.5,tilt=-0.5,width=1.0))
    assert_almost_equal(activations.bicentral_activation(0.0,lower=0.5,tilt=0.0,width=0.5),(7777/9077))
    assert_almost_equal(activations.bicentral_activation(-0.5,lower=0.5,tilt=0.0,width=1.0),
                        activations.bicentral_activation(0.5,lower=0.5,tilt=0.0,width=1.0))
    assert_almost_equal(activations.bicentral_activation(0.0,lower=0.0,tilt=1.0,width=0.5),
                        activations.bicentral_activation(0.0,lower=0.0,tilt=-1.0,width=0.5))
    assert_almost_equal(activations.bicentral_activation(0.0,lower=0.25,tilt=0.0,width=0.5),(7956/8131))
    assert_almost_equal(activations.bicentral_activation(0.0,lower=0.0,tilt=0.5,width=0.5),
                        activations.bicentral_activation(0.0,lower=0.0,tilt=-0.5,width=0.5))
    assert_almost_equal(activations.bicentral_activation(0.0,lower=0.25,tilt=0.0,width=1.0),(5003/5004))
    assert_almost_equal(activations.bicentral_activation(-1.0,lower=0.75,tilt=1.0,width=0.5),
                        activations.bicentral_activation(1.0,lower=0.75,tilt=-1.0,width=0.5))
    assert_almost_equal(activations.bicentral_activation(-1.0,lower=0.5,tilt=0.5,width=0.5),
                        activations.bicentral_activation(1.0,lower=0.5,tilt=-0.5,width=0.5))
    assert_almost_equal(activations.bicentral_activation(-1.0,lower=0.5,tilt=0.0,width=0.0),
                        activations.bicentral_activation(1.0,lower=0.5,tilt=0.0,width=0.0))
    assert_almost_equal(activations.bicentral_activation(-1.0,lower=1.0,tilt=1.0,width=0.5),
                        activations.bicentral_activation(1.0,lower=1.0,tilt=-1.0,width=0.5))
    assert_almost_equal(activations.bicentral_activation(-1.0,lower=0.5,tilt=0.0,width=0.25),
                        activations.bicentral_activation(1.0,lower=0.5,tilt=0.0,width=0.25))
    assert_almost_equal(activations.bicentral_activation(-1.0,lower=1.0,tilt=0.5,width=0.5),
                        activations.bicentral_activation(1.0,lower=1.0,tilt=-0.5,width=0.5))
    assert_almost_equal(activations.bicentral_activation(-0.5,lower=0.5,tilt=0.0,width=0.0),
                        activations.bicentral_activation(0.5,lower=0.5,tilt=0.0,width=0.0))
    assert_almost_equal(activations.bicentral_activation(-1.0,lower=0.5,tilt=0.0,width=0.5),
                        activations.bicentral_activation(1.0,lower=0.5,tilt=0.0,width=0.5))
    assert_almost_equal(activations.bicentral_activation(1.0,lower=0.0,tilt=1.0,width=0.5),
                        activations.bicentral_activation(-1.0,lower=0.0,tilt=-1.0,width=0.5))
    assert_almost_equal(activations.bicentral_activation(-1.0,lower=0.75,tilt=0.0,width=0.0),
                        activations.bicentral_activation(1.0,lower=0.75,tilt=0.0,width=0.0))
    assert_almost_equal(activations.bicentral_activation(-1.0,lower=1.0,tilt=0.0,width=0.0),
                        activations.bicentral_activation(1.0,lower=1.0,tilt=0.0,width=0.0))
    assert_almost_equal(activations.bicentral_activation(-0.5,lower=1.0,tilt=0.0,width=0.0),
                        activations.bicentral_activation(0.5,lower=1.0,tilt=0.0,width=0.0))
    assert_almost_equal(activations.bicentral_activation(-1.0,lower=1.0,tilt=0.0,width=0.25),
                        activations.bicentral_activation(1.0,lower=1.0,tilt=0.0,width=0.25))
    assert_almost_equal(activations.bicentral_activation(-1.0,lower=0.5,tilt=0.0,width=0.75),
                        activations.bicentral_activation(1.0,lower=0.5,tilt=0.0,width=0.75))
    assert_almost_equal(activations.bicentral_activation(-1.0,lower=1.0,tilt=0.0,width=0.5),
                        activations.bicentral_activation(1.0,lower=1.0,tilt=0.0,width=0.5))
    assert_almost_equal(activations.bicentral_activation(1.0,lower=0.5,tilt=0.5,width=0.0),
                        activations.bicentral_activation(-1.0,lower=0.5,tilt=-0.5,width=0.0))
    assert_almost_equal(activations.bicentral_activation(-1.0,lower=0.75,tilt=0.0,width=0.5),
                        activations.bicentral_activation(1.0,lower=0.75,tilt=0.0,width=0.5))
    assert_almost_equal(activations.bicentral_activation(-1.0,lower=1.0,tilt=0.0,width=0.75),
                        activations.bicentral_activation(1.0,lower=1.0,tilt=0.0,width=0.75))
    assert_almost_equal(activations.bicentral_activation(-0.5,lower=1.0,tilt=0.0,width=0.5),
                        activations.bicentral_activation(0.5,lower=1.0,tilt=0.0,width=0.5))
    assert_almost_equal(activations.bicentral_activation(-1.0,lower=1.0,tilt=0.0,width=1.0),
                        activations.bicentral_activation(1.0,lower=1.0,tilt=0.0,width=1.0))
    assert_almost_equal(activations.bicentral_activation(0.0,lower=1.0,tilt=0.5,width=0.5),
                        activations.bicentral_activation(0.0,lower=1.0,tilt=-0.5,width=0.5))
    assert_almost_equal(activations.bicentral_activation(1.0,lower=1.0,tilt=0.5,width=0.5),
                        activations.bicentral_activation(-1.0,lower=1.0,tilt=-0.5,width=0.5))
    assert_almost_equal(activations.bicentral_activation(-0.5,lower=1.0,tilt=0.0,width=1.0),
                        activations.bicentral_activation(0.5,lower=1.0,tilt=0.0,width=1.0))
    assert_almost_equal(activations.bicentral_activation(-0.5,lower=1.0,tilt=1.0,width=0.5),
                        activations.bicentral_activation(0.5,lower=1.0,tilt=-1.0,width=0.5))
    assert_almost_equal(activations.bicentral_activation(1.0,lower=0.5,tilt=1.0,width=0.0),
                        activations.bicentral_activation(-1.0,lower=0.5,tilt=-1.0,width=0.0))
    assert_almost_equal(activations.bicentral_activation(-0.5,lower=0.5,tilt=1.0,width=0.5),
                        activations.bicentral_activation(0.5,lower=0.5,tilt=-1.0,width=0.5))
    assert_almost_equal(activations.bicentral_activation(-1.0,lower=0.75,tilt=0.0,width=1.0),
                        activations.bicentral_activation(1.0,lower=0.75,tilt=0.0,width=1.0))
    assert_almost_equal(activations.bicentral_activation(1.0,lower=0.5,tilt=1.0,width=0.25),
                        activations.bicentral_activation(-1.0,lower=0.5,tilt=-1.0,width=0.25))
    assert_almost_equal(activations.bicentral_activation(-1.0,lower=0.5,tilt=1.0,width=1.0),
                        activations.bicentral_activation(1.0,lower=0.5,tilt=-1.0,width=1.0))
    assert_almost_equal(activations.bicentral_activation(1.0,lower=0.5,tilt=1.0,width=0.75),
                        activations.bicentral_activation(-1.0,lower=0.5,tilt=-1.0,width=0.75))
    assert_almost_equal(activations.bicentral_activation(-1.0,lower=0.5,tilt=0.5,width=1.0),
                        activations.bicentral_activation(1.0,lower=0.5,tilt=-0.5,width=1.0))
    assert_almost_equal(activations.bicentral_activation(0.0,lower=1.0,tilt=1.0,width=0.5),
                        activations.bicentral_activation(0.0,lower=1.0,tilt=-1.0,width=0.5))
    assert_almost_equal(activations.bicentral_activation(1.0,lower=0.75,tilt=1.0,width=0.5),
                        activations.bicentral_activation(-1.0,lower=0.75,tilt=-1.0,width=0.5))
    assert_almost_equal(activations.bicentral_activation(1.0,lower=1.0,tilt=1.0,width=0.5),
                        activations.bicentral_activation(-1.0,lower=1.0,tilt=-1.0,width=0.5))
    assert_almost_equal(activations.bicentral_activation(-1.0,lower=0.5,tilt=0.0,width=1.0),
                        activations.bicentral_activation(1.0,lower=0.5,tilt=0.0,width=1.0))
    assert_almost_equal(activations.bicentral_activation(0.0,lower=0.75,tilt=1.0,width=0.5),
                        activations.bicentral_activation(0.0,lower=0.75,tilt=-1.0,width=0.5))
    assert_almost_equal(activations.bicentral_activation(0.0,lower=0.5,tilt=1.0,width=0.25),
                        activations.bicentral_activation(0.0,lower=0.5,tilt=-1.0,width=0.25))
    assert_almost_equal(activations.bicentral_activation(0.0,lower=0.5,tilt=1.0,width=0.75),
                        activations.bicentral_activation(0.0,lower=0.5,tilt=-1.0,width=0.75))
    assert_almost_equal(activations.bicentral_activation(0.0,lower=0.5,tilt=1.0,width=1.0),
                        activations.bicentral_activation(0.0,lower=0.5,tilt=-1.0,width=1.0))
    assert_almost_equal(activations.bicentral_activation(-0.5,lower=0.5,tilt=1.0,width=1.0),
                        activations.bicentral_activation(0.5,lower=0.5,tilt=-1.0,width=1.0))
    assert_almost_equal(activations.bicentral_activation(0.0,lower=0.5,tilt=0.5,width=0.5),
                        activations.bicentral_activation(0.0,lower=0.5,tilt=-0.5,width=0.5))
    assert_almost_equal(activations.bicentral_activation(0.0,lower=0.25,tilt=1.0,width=0.5),
                        activations.bicentral_activation(0.0,lower=0.25,tilt=-1.0,width=0.5))
    assert_almost_equal(activations.bicentral_activation(0.0,lower=0.5,tilt=0.5,width=1.0),
                        activations.bicentral_activation(0.0,lower=0.5,tilt=-0.5,width=1.0))
    assert_almost_equal(activations.bicentral_activation(-0.5,lower=0.5,tilt=0.0,width=1.0),
                        activations.bicentral_activation(0.5,lower=0.5,tilt=0.0,width=1.0))
    assert_almost_equal(activations.bicentral_activation(0.0,lower=0.0,tilt=1.0,width=0.5),
                        activations.bicentral_activation(0.0,lower=0.0,tilt=-1.0,width=0.5))
    assert_almost_equal(activations.bicentral_activation(0.0,lower=0.0,tilt=0.5,width=0.5),
                        activations.bicentral_activation(0.0,lower=0.0,tilt=-0.5,width=0.5))
    assert_almost_equal(activations.bicentral_activation(1.0,lower=0.0,tilt=0.5,width=0.5),
                        activations.bicentral_activation(-1.0,lower=0.0,tilt=-0.5,width=0.5))
    assert_almost_equal(activations.bicentral_activation(1.0,lower=0.25,tilt=1.0,width=0.5),
                        activations.bicentral_activation(-1.0,lower=0.25,tilt=-1.0,width=0.5))
    assert_almost_equal(activations.bicentral_activation(-1.0,lower=0.5,tilt=-1.0,width=0.5),
                        activations.bicentral_activation(-0.5,lower=0.5,tilt=-1.0,width=0.0))
    assert_almost_equal(activations.bicentral_activation(1.0,lower=0.5,tilt=1.0,width=0.5),
                        activations.bicentral_activation(0.5,lower=0.5,tilt=1.0,width=0.0))
    assert_almost_equal(activations.bicentral_activation(0.5,lower=1.0,tilt=1.0,width=0.5),
                        activations.bicentral_activation(-0.5,lower=1.0,tilt=-1.0,width=0.5))
    assert_almost_equal(activations.bicentral_activation(0.0,lower=0.5,tilt=-1.0,width=0.5),
                        activations.bicentral_activation(-0.5,lower=0.5,tilt=-1.0,width=1.0))
    assert_almost_equal(activations.bicentral_activation(0.0,lower=0.5,tilt=1.0,width=0.5),
                        activations.bicentral_activation(0.5,lower=0.5,tilt=1.0,width=1.0))

def test_fourth_square_abs():
    for curve in [0.0,0.25,0.5,0.75,1.0]:
        assert activations.fourth_square_abs_activation(0.0,curve) == 0.0
        assert activations.fourth_square_abs_activation(1.0,curve) == 1.0
        assert activations.fourth_square_abs_activation(-1.0,curve) == 1.0
    assert activations.fourth_square_abs_activation(-0.5,1.0) == 0.0625
    assert activations.fourth_square_abs_activation(0.5,1.0) == 0.0625
    assert activations.fourth_square_abs_activation(-0.5,0.75) == 0.15625
    assert activations.fourth_square_abs_activation(0.5,0.75) == 0.15625
    assert activations.fourth_square_abs_activation(-0.75,0.5) == 0.5625
    assert activations.fourth_square_abs_activation(-0.5,0.5) == 0.25
    assert activations.fourth_square_abs_activation(-0.25,0.5) == 0.0625
    assert activations.fourth_square_abs_activation(0.25,0.5) == 0.0625
    assert activations.fourth_square_abs_activation(0.5,0.5) == 0.25
    assert activations.fourth_square_abs_activation(0.75,0.5) == 0.5625
    assert activations.fourth_square_abs_activation(-0.5,0.25) == 0.375
    assert activations.fourth_square_abs_activation(0.5,0.25) == 0.375
    assert activations.fourth_square_abs_activation(-0.5,0.0) == 0.5
    assert activations.fourth_square_abs_activation(0.5,0.0) == 0.5

def test_mexican_hat():
    for lower in [0.0, 0.25, 0.5, 0.75, 1.0]:
        for width in [0.0, 0.25, 0.5, 0.75, 1.0]:
            assert activations.mexican_hat_activation(0.0,
                                                      lower=lower,
                                                      width=width) == 1.0
    assert activations.mexican_hat_activation(-1.0,lower=0.0,width=0.0) == 0.0
    assert activations.mexican_hat_activation(-1.0,lower=0.0,width=0.25) == 0.0
    assert activations.mexican_hat_activation(-1.0,lower=0.0,width=0.5) == 0.0
    assert activations.mexican_hat_activation(-1.0,lower=0.0,width=0.75) == 0.0
    assert activations.mexican_hat_activation(-1.0,lower=0.0,width=1.0) == 0.0
    assert activations.mexican_hat_activation(-1.0,lower=0.25,width=0.0) == 0.0
    assert activations.mexican_hat_activation(-1.0,lower=0.25,width=0.5) == 0.0
    assert activations.mexican_hat_activation(-1.0,lower=0.25,width=1.0) == -0.15625
    assert activations.mexican_hat_activation(-1.0,lower=0.5,width=0.0) == 0.0
    assert activations.mexican_hat_activation(-1.0,lower=0.5,width=0.25) == 0.0
    assert activations.mexican_hat_activation(-1.0,lower=0.5,width=0.5) == -0.1875
    assert activations.mexican_hat_activation(-1.0,lower=0.5,width=0.75) == -0.1875
    assert activations.mexican_hat_activation(-1.0,lower=0.5,width=1.0) == -0.1875
    assert activations.mexican_hat_activation(-1.0,lower=0.75,width=0.0) == 0.0
    assert activations.mexican_hat_activation(-1.0,lower=0.75,width=0.5) == -0.21875
    assert activations.mexican_hat_activation(-1.0,lower=0.75,width=1.0) == -0.21875
    assert activations.mexican_hat_activation(-1.0,lower=1.0,width=0.0) == 0.0
    assert activations.mexican_hat_activation(-1.0,lower=1.0,width=0.25) == -0.25
    assert activations.mexican_hat_activation(-1.0,lower=1.0,width=0.5) == -0.25
    assert activations.mexican_hat_activation(-1.0,lower=1.0,width=0.75) == -0.25
    assert activations.mexican_hat_activation(-1.0,lower=1.0,width=1.0) == -0.25
    assert activations.mexican_hat_activation(-0.5,lower=0.0,width=0.0) == 0.0
    assert activations.mexican_hat_activation(-0.5,lower=0.0,width=0.5) == 0.875
    assert activations.mexican_hat_activation(-0.5,lower=0.0,width=1.0) == 1.0
    assert activations.mexican_hat_activation(-0.5,lower=0.5,width=0.0) == -0.1875
    assert activations.mexican_hat_activation(-0.5,lower=0.5,width=0.5) == 0.8125
    assert activations.mexican_hat_activation(-0.5,lower=0.5,width=1.0) == 1.0
    assert activations.mexican_hat_activation(-0.5,lower=1.0,width=0.0) == -0.25
    assert activations.mexican_hat_activation(-0.5,lower=1.0,width=0.5) == 0.75
    assert activations.mexican_hat_activation(-0.5,lower=1.0,width=1.0) == 1.0
    assert activations.mexican_hat_activation(0.5,lower=0.0,width=0.0) == 0.0
    assert activations.mexican_hat_activation(0.5,lower=0.0,width=0.5) == 0.875
    assert activations.mexican_hat_activation(0.5,lower=0.0,width=1.0) == 1.0
    assert activations.mexican_hat_activation(0.5,lower=0.5,width=0.0) == -0.1875
    assert activations.mexican_hat_activation(0.5,lower=0.5,width=0.5) == 0.8125
    assert activations.mexican_hat_activation(0.5,lower=0.5,width=1.0) == 1.0
    assert activations.mexican_hat_activation(0.5,lower=1.0,width=0.0) == -0.25
    assert activations.mexican_hat_activation(0.5,lower=1.0,width=0.5) == 0.75
    assert activations.mexican_hat_activation(0.5,lower=1.0,width=1.0) == 1.0
    assert activations.mexican_hat_activation(1.0,lower=0.0,width=0.0) == 0.0
    assert activations.mexican_hat_activation(1.0,lower=0.0,width=0.25) == 0.0
    assert activations.mexican_hat_activation(1.0,lower=0.0,width=0.5) == 0.0
    assert activations.mexican_hat_activation(1.0,lower=0.0,width=0.75) == 0.0
    assert activations.mexican_hat_activation(1.0,lower=0.0,width=1.0) == 0.0
    assert activations.mexican_hat_activation(1.0,lower=0.25,width=0.0) == 0.0
    assert activations.mexican_hat_activation(1.0,lower=0.25,width=0.5) == 0.0
    assert activations.mexican_hat_activation(1.0,lower=0.25,width=1.0) == -0.15625
    assert activations.mexican_hat_activation(1.0,lower=0.5,width=0.0) == 0.0
    assert activations.mexican_hat_activation(1.0,lower=0.5,width=0.25) == 0.0
    assert activations.mexican_hat_activation(1.0,lower=0.5,width=0.5) == -0.1875
    assert activations.mexican_hat_activation(1.0,lower=0.5,width=0.75) == -0.1875
    assert activations.mexican_hat_activation(1.0,lower=0.5,width=1.0) == -0.1875
    assert activations.mexican_hat_activation(1.0,lower=0.75,width=0.0) == 0.0
    assert activations.mexican_hat_activation(1.0,lower=0.75,width=0.5) == -0.21875
    assert activations.mexican_hat_activation(1.0,lower=0.75,width=1.0) == -0.21875
    assert activations.mexican_hat_activation(1.0,lower=1.0,width=0.0) == 0.0
    assert activations.mexican_hat_activation(1.0,lower=1.0,width=0.25) == -0.25
    assert activations.mexican_hat_activation(1.0,lower=1.0,width=0.5) == -0.25
    assert activations.mexican_hat_activation(1.0,lower=1.0,width=0.75) == -0.25
    assert activations.mexican_hat_activation(1.0,lower=1.0,width=1.0) == -0.25

def test_rational_quadratic():
    for lower in [0.0, 0.25, 0.5, 0.75, 1.0]:
        for width in [0.0, 0.25, 0.5, 0.75, 1.0]:
            assert activations.rational_quadratic_activation(0.0,
                                                             lower=lower,
                                                             width=width) == 1.0
    assert 0.0 <= activations.rational_quadratic_activation(-0.5,lower=0.5,width=0.0) <= 0.000123
    assert 0.0 <= activations.rational_quadratic_activation(-1.0,lower=0.5,width=0.0) <= 0.000123
    assert 0.0 <= activations.rational_quadratic_activation(-1.0,lower=0.75,width=0.5) <= 0.000123
    assert 0.0 <= activations.rational_quadratic_activation(-1.0,lower=1.0,width=0.75) <= 0.000123
    assert 0.0 <= activations.rational_quadratic_activation(0.5,lower=0.5,width=0.0) <= 0.000123
    assert 0.0 <= activations.rational_quadratic_activation(1.0,lower=0.5,width=0.0) <= 0.000123
    assert 0.0 <= activations.rational_quadratic_activation(1.0,lower=0.75,width=0.5) <= 0.000123
    assert 0.0 <= activations.rational_quadratic_activation(1.0,lower=1.0,width=0.75) <= 0.000123
    assert activations.rational_quadratic_activation(-0.5,lower=0.5,width=1.0) == 0.8
    assert activations.rational_quadratic_activation(-1.0,lower=0.5,width=1.0) == 0.5
    assert activations.rational_quadratic_activation(0.5,lower=0.5,width=1.0) == 0.8
    assert activations.rational_quadratic_activation(1.0,lower=0.5,width=1.0) == 0.5
    assert_almost_equal(activations.rational_quadratic_activation(-1.0,lower=0.5,width=0.25),
                        activations.rational_quadratic_activation(1.0,lower=0.5,width=0.25))
    assert_almost_equal(activations.rational_quadratic_activation(-1.0,lower=0.5,width=0.5),
                        activations.rational_quadratic_activation(1.0,lower=0.5,width=0.5))
    assert_almost_equal(activations.rational_quadratic_activation(-0.5,lower=0.5,width=0.5),
                        activations.rational_quadratic_activation(0.5,lower=0.5,width=0.5))
    assert_almost_equal(activations.rational_quadratic_activation(-1.0,lower=0.5,width=0.75),
                        activations.rational_quadratic_activation(1.0,lower=0.5,width=0.75))
    assert_almost_equal(activations.rational_quadratic_activation(-1.0,lower=0.25,width=0.0),
                        activations.rational_quadratic_activation(1.0,lower=0.25,width=0.0))
    assert_almost_equal(activations.rational_quadratic_activation(-1.0,lower=0.25,width=0.5),
                        activations.rational_quadratic_activation(1.0,lower=0.25,width=0.5))
    assert_almost_equal(activations.rational_quadratic_activation(-1.0,lower=1.0,width=1.0),
                        activations.rational_quadratic_activation(1.0,lower=1.0,width=1.0))
    assert_almost_equal(activations.rational_quadratic_activation(-1.0,lower=0.75,width=1.0),
                        activations.rational_quadratic_activation(1.0,lower=0.75,width=1.0))
    assert_almost_equal(activations.rational_quadratic_activation(-1.0,lower=0.0,width=0.0),
                        activations.rational_quadratic_activation(1.0,lower=0.0,width=0.0))
    assert_almost_equal(activations.rational_quadratic_activation(-0.5,lower=0.0,width=0.0),
                        activations.rational_quadratic_activation(0.5,lower=0.0,width=0.0))
    assert_almost_equal(activations.rational_quadratic_activation(-1.0,lower=0.0,width=0.25),
                        activations.rational_quadratic_activation(1.0,lower=0.0,width=0.25))
    assert_almost_equal(activations.rational_quadratic_activation(-1.0,lower=0.0,width=0.5),
                        activations.rational_quadratic_activation(1.0,lower=0.0,width=0.5))
    assert_almost_equal(activations.rational_quadratic_activation(-1.0,lower=0.25,width=1.0),
                        activations.rational_quadratic_activation(1.0,lower=0.25,width=1.0))
    assert_almost_equal(activations.rational_quadratic_activation(-0.5,lower=0.0,width=0.5),
                        activations.rational_quadratic_activation(0.5,lower=0.0,width=0.5))
    assert_almost_equal(activations.rational_quadratic_activation(-1.0,lower=0.0,width=0.75),
                        activations.rational_quadratic_activation(1.0,lower=0.0,width=0.75))
    assert_almost_equal(activations.rational_quadratic_activation(-0.5,lower=1.0,width=1.0),
                        activations.rational_quadratic_activation(0.5,lower=1.0,width=1.0))
    assert_almost_equal(activations.rational_quadratic_activation(-1.0,lower=0.0,width=1.0),
                        activations.rational_quadratic_activation(1.0,lower=0.0,width=1.0))
    assert_almost_equal(activations.rational_quadratic_activation(-0.5,lower=0.0,width=1.0),
                        activations.rational_quadratic_activation(0.5,lower=0.0,width=1.0))
    assert_almost_equal(activations.rational_quadratic_activation(-1.0,lower=0.5,width=0.25),
                        activations.rational_quadratic_activation(1.0,lower=0.5,width=0.25))
    assert_almost_equal(activations.rational_quadratic_activation(-1.0,lower=0.5,width=0.5),
                        activations.rational_quadratic_activation(1.0,lower=0.5,width=0.5))
    assert_almost_equal(activations.rational_quadratic_activation(-0.5,lower=0.5,width=0.5),
                        activations.rational_quadratic_activation(0.5,lower=0.5,width=0.5))
    assert_almost_equal(activations.rational_quadratic_activation(-1.0,lower=0.5,width=0.75),
                        activations.rational_quadratic_activation(1.0,lower=0.5,width=0.75))
    assert_almost_equal(activations.rational_quadratic_activation(-1.0,lower=0.25,width=0.0),
                        activations.rational_quadratic_activation(1.0,lower=0.25,width=0.0))
    assert_almost_equal(activations.rational_quadratic_activation(-1.0,lower=0.25,width=0.5),
                        activations.rational_quadratic_activation(1.0,lower=0.25,width=0.5))
    assert_almost_equal(activations.rational_quadratic_activation(-1.0,lower=1.0,width=1.0),
                        activations.rational_quadratic_activation(1.0,lower=1.0,width=1.0))
    assert_almost_equal(activations.rational_quadratic_activation(-1.0,lower=0.75,width=1.0),
                        activations.rational_quadratic_activation(1.0,lower=0.75,width=1.0))
    assert_almost_equal(activations.rational_quadratic_activation(-1.0,lower=0.0,width=0.0),
                        activations.rational_quadratic_activation(1.0,lower=0.0,width=0.0))
    assert_almost_equal(activations.rational_quadratic_activation(-0.5,lower=0.0,width=0.0),
                        activations.rational_quadratic_activation(0.5,lower=0.0,width=0.0))
    assert_almost_equal(activations.rational_quadratic_activation(-1.0,lower=0.0,width=0.25),
                        activations.rational_quadratic_activation(1.0,lower=0.0,width=0.25))
    assert_almost_equal(activations.rational_quadratic_activation(-1.0,lower=0.0,width=0.5),
                        activations.rational_quadratic_activation(1.0,lower=0.0,width=0.5))
    assert_almost_equal(activations.rational_quadratic_activation(-1.0,lower=0.25,width=1.0),
                        activations.rational_quadratic_activation(1.0,lower=0.25,width=1.0))
    assert_almost_equal(activations.rational_quadratic_activation(-0.5,lower=0.0,width=0.5),
                        activations.rational_quadratic_activation(0.5,lower=0.0,width=0.5))
    assert_almost_equal(activations.rational_quadratic_activation(-1.0,lower=0.0,width=0.75),
                        activations.rational_quadratic_activation(1.0,lower=0.0,width=0.75))
    assert_almost_equal(activations.rational_quadratic_activation(-0.5,lower=1.0,width=1.0),
                        activations.rational_quadratic_activation(0.5,lower=1.0,width=1.0))
    assert_almost_equal(activations.rational_quadratic_activation(-1.0,lower=0.0,width=1.0),
                        activations.rational_quadratic_activation(1.0,lower=0.0,width=1.0))
    assert_almost_equal(activations.rational_quadratic_activation(-0.5,lower=0.0,width=1.0),
                        activations.rational_quadratic_activation(0.5,lower=0.0,width=1.0))


def plus_activation(x):
    """ Not useful - just a check. """
    return abs(x+1)

def test_add_plus():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'test_configuration')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    config.genome_config.add_activation('plus', plus_activation)
    assert config.genome_config.activation_defs.get('plus') is not None
    assert config.genome_config.activation_defs.is_valid('plus')

def multiparam_plus_activation(x, a):
    """Again, not useful..."""
    return abs(x+a)

def test_add_multiparam_plus():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'test_configuration')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    config.genome_config.add_activation('multiparam_plus', multiparam_plus_activation,
                                        a={'min_value':-1.0, 'max_value':3.0})
    assert config.genome_config.multiparameterset.get_MPF('multiparam_plus',
                                                          'activation') is not None
    assert config.genome_config.activation_defs.is_valid('multiparam_plus')

NORM_ACT_FUNC = """sigmoid tanh sigmoid_approx tanh_approx sin gauss relu identity clamped inv
                   log expanded_log skewed_log1p log1p exp abs hat square cube square_wave
                   triangle_wave rectangular""".split()
MULT_ACT_FUNC = """multiparam_relu multiparam_relu_softplus multiparam_elu weighted_lu
                   clamped_tanh_step clamped_step clamped_log1p_step multiparam_sigmoid
                   hat_gauss_rectangular scaled_expanded_log multiparam_log_inv scaled_log1p
                   multiparam_tanh_log1p multiparam_pow wave multiparam_tanh_approx bicentral
                   multiparam_sigmoid_approx multiparam_gauss fourth_square_abs
                   mexican_hat rational_quadratic""".split()

def test_function_set():
    m = multiparameter.MultiParameterSet('activation')
    s = activations.ActivationFunctionSet(m)
    for name in NORM_ACT_FUNC:
        assert s.get(name) is not None, "Failed to get {0}".format(name)
        assert s.is_valid(name), "Function {0} not valid".format(name)
    for name in MULT_ACT_FUNC:
        assert m.get_MPF(name, 'activation') is not None, "Failed get_MPF({0})".format(name)
        assert s.is_valid(name), "Function {0} not valid".format(name)

    assert not s.is_valid('foo')

def test_bad_add1():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'test_configuration')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    
    try:
        config.genome_config.add_activation('1.0',1.0)
    except TypeError:
        pass
    else:
        raise Exception("Should have had a TypeError/derived for 'function' 1.0")

def dud_function():
    return 0.0

def test_bad_add2():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'test_configuration')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    
    try:
        config.genome_config.add_activation('dud_function',dud_function)
    except TypeError:
        pass
    else:
        raise Exception("Should have had a TypeError/derived for dud_function")

def test_get_MPF():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'test_configuration')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    for func in MULT_ACT_FUNC:
        assert config.genome_config.get_activation_MPF(func) is not None

    try:
        ignored = config.genome_config.get_activation_MPF('foo')
    except LookupError:
        pass
    else:
        raise Exception("Should have had a LookupError/derived for get_activation_MPF 'foo'")

def test_get_Evolved_MPF_simple():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'test_configuration')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    param_namespace=Namespace()

    for func in MULT_ACT_FUNC:
        assert config.genome_config.get_activation_Evolved_MPF(func,
                                                               param_namespace=param_namespace) is not None

    try:
        ignored = config.genome_config.get_activation_Evolved_MPF('foo')
    except LookupError:
        pass
    else:
        raise Exception("Should have had a LookupError/derived for get_activation_Evolved_MPF 'foo'")

def test_get_Evolved_MPF_complex():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'test_configuration')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    param_namespace=Namespace()

    assert config.genome_config.get_activation_Evolved_MPF('multiparam_relu(0.5)',
                                                           param_namespace=param_namespace) is not None
    assert config.genome_config.get_activation_Evolved_MPF('multiparam_relu_softplus(0.5,0.5)',
                                                           param_namespace=param_namespace) is not None
    assert config.genome_config.get_activation_Evolved_MPF('multiparam_elu(0.5,0.5)',
                                                           param_namespace=param_namespace) is not None
    assert config.genome_config.get_activation_Evolved_MPF('weighted_lu(0.5,0.5)',
                                                           param_namespace=param_namespace) is not None
    assert config.genome_config.get_activation_Evolved_MPF('clamped_tanh_step(0.5)',
                                                           param_namespace=param_namespace) is not None
    assert config.genome_config.get_activation_Evolved_MPF('multiparam_sigmoid(0.5)',
                                                           param_namespace=param_namespace) is not None

    try:
        ignored = config.genome_config.get_activation_Evolved_MPF('foo(0.5)')
    except LookupError:
        pass
    else:
        raise Exception("Should have had a LookupError/derived for get_activation_Evolved_MPF 'foo(0.5)'")

    try:
        ignored = config.genome_config.get_activation_Evolved_MPF('multiparam_relu(0.5,0.5,0.5)')
    except RuntimeError:
        pass
    else:
        raise Exception("Should have had a RuntimeError/derived for get_activation_Evolved_MPF 'multiparam_relu(0.5,0.5,0.5)'")

    test_result = config.genome_config.get_activation_Evolved_MPF('multiparam_relu(0.5)',
                                                                  param_namespace=param_namespace)
    assert str(test_result) == str(config.genome_config.get_activation_Evolved_MPF(str(test_result),
                                                                                   param_namespace=param_namespace))
    partial_func = config.genome_config.multiparameterset.get_func(str(test_result), 'activation')
    assert partial_func is not None
    extracted = repr_util.repr_extract_function_name(partial_func, no_match=repr_util.ERR_IF_NO_MATCH)
    assert extracted is not None
    assert '0.5' in extracted, "Wrong extracted {0!r} from partial_func {1!r}".format(extracted,
                                                                                      partial_func)
    assert 'multiparam_relu_activation' in extracted, "Wrong extracted {0!r} from partial_func {1!r}".format(extracted,
                                                                                                             partial_func)
    extracted2 = repr_util.repr_extract_function_name(partial_func, with_module=False, as_partial=False, OK_with_args=True)
    assert extracted2 == 'multiparam_relu(z,0.5)', "Wrong extracted2 {0!r} from partial_func {1!r}".format(extracted2,
                                                                                                           partial_func)

if __name__ == '__main__':
    test_sigmoid()
    test_tanh()
    test_sigmoid_approx()
    test_tanh_approx()
    test_sin()
    test_gauss()
    test_relu()
    test_softplus()
    test_identity()
    test_clamped()
    test_inv()
    test_log()
    test_expanded_log()
    test_skewed_log1p()
    test_log1p()
    test_exp()
    test_abs()
    test_hat()
    test_square()
    test_cube()
    test_square_wave()
    test_triangle_wave()
    test_rectangular()
    test_multiparam_relu()
    test_multiparam_elu()
    test_weighted_lu()
    test_multiparam_relu_softplus()
    test_clamped_tanh_step()
    test_multiparam_sigmoid()
    test_clamped_step()
    test_hat_gauss_rectangular()
    test_scaled_expanded_log()
    test_multiparam_log_inv()
    test_scaled_log1p()
    test_multiparam_tanh_log1p()
    test_multiparam_pow()
    test_wave()
    test_multiparam_tanh_approx()
    test_multiparam_sigmoid_approx()
    test_multiparam_gauss()
    test_bicentral()
    test_fourth_square_abs()
    test_mexican_hat()
    test_add_plus()
    test_add_multiparam_plus()
    test_function_set()
    test_get_MPF()
    test_get_Evolved_MPF_simple()
    test_get_Evolved_MPF_complex()
    test_bad_add1()
    test_bad_add2()
