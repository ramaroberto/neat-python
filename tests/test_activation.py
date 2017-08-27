from __future__ import print_function, division
import os
import math
import sys
import warnings

import neat
from neat import activations, multiparameter, repr_util

warnings.simplefilter('default')

# TODO: These tests are just smoke tests to make sure nothing has become badly broken.  Expand
# to include more detailed tests of actual functionality.

class NotAlmostEqualException(AssertionError):
    pass


def assert_almost_equal(a, b):
    if abs(a - b) > 1e-6:
        max_abs = max(abs(a), abs(b))
        abs_rel_err = abs(a - b) / max_abs
        if abs_rel_err > 1e-6:
            raise NotAlmostEqualException("{0!r} !~= {1!r}".format(float(a), float(b)))

def assert_inv_func_adds_to(f, a, b):
    if isinstance(a, float):
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
    assert_almost_equal(activations.softplus_activation(sys.float_info.max),0.0)
    assert_almost_equal((activations.softplus_activation(1.0)
                         -activations.softplus_activation(-1.0)),1.0)
    assert_almost_equal(activations.softplus_activation(5.0),5.0)


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


def test_abs():
    assert activations.abs_activation(-1.0) == 1.0
    assert activations.abs_activation(0.0) == 0.0
    assert activations.abs_activation(-1.0) == 1.0


def test_hat():
    assert activations.hat_activation(-1.0) == 0.0
    assert activations.hat_activation(0.0) == 1.0
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

def dud_function():
    return 0.0

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
    assert activations.multiparam_elu_activation(0.0,1.0,1.0) == 0.0
    assert activations.multiparam_elu_activation(0.0,1.0,0.5) == 0.0
    assert activations.multiparam_elu_activation(0.0,1.0,0.0) == 0.0
    assert activations.multiparam_elu_activation(0.0,1.0,-0.5) == 0.0
    assert activations.multiparam_elu_activation(0.0,1.0,-1.0) == 0.0
    assert activations.multiparam_elu_activation(0.0,0.75,1.0) == 0.0
    assert activations.multiparam_elu_activation(0.0,0.75,0.0) == 0.0
    assert activations.multiparam_elu_activation(0.0,0.75,-1.0) == 0.0
    assert activations.multiparam_elu_activation(0.0,0.5,1.0) == 0.0
    assert activations.multiparam_elu_activation(0.0,0.5,0.5) == 0.0
    assert activations.multiparam_elu_activation(0.0,0.5,0.0) == 0.0
    assert activations.multiparam_elu_activation(0.0,0.5,-0.5) == 0.0
    assert activations.multiparam_elu_activation(0.0,0.5,-1.0) == 0.0
    assert activations.multiparam_elu_activation(0.0,0.25,1.0) == 0.0
    assert activations.multiparam_elu_activation(0.0,0.25,0.0) == 0.0
    assert activations.multiparam_elu_activation(0.0,0.25,-1.0) == 0.0
    assert activations.multiparam_elu_activation(0.0,0.0,1.0) == 0.0
    assert activations.multiparam_elu_activation(0.5,0.0,1.0) == 0.5
    assert activations.multiparam_elu_activation(1.0,0.0,1.0) == 1.0
    assert activations.multiparam_elu_activation(0.0,0.0,0.5) == 0.0
    assert activations.multiparam_elu_activation(1.0,0.0,0.5) == 1.0
    assert activations.multiparam_elu_activation(0.0,0.0,0.0) == 0.0
    assert activations.multiparam_elu_activation(0.5,0.0,0.0) == 0.5
    assert activations.multiparam_elu_activation(1.0,0.0,0.0) == 1.0
    assert activations.multiparam_elu_activation(0.0,0.0,-0.5) == 0.0
    assert activations.multiparam_elu_activation(1.0,0.0,-0.5) == 1.0
    assert activations.multiparam_elu_activation(0.0,0.0,-1.0) == 0.0
    assert activations.multiparam_elu_activation(0.5,0.0,-1.0) == 0.5
    assert activations.multiparam_elu_activation(1.0,0.0,-1.0) == 1.0
    assert_almost_equal(activations.multiparam_elu_activation(-0.5,1.0,1.0),
                        activations.multiparam_elu_activation(-0.5,1.0,0.0))
    assert_almost_equal(activations.multiparam_elu_activation(-60.0,0.5,0.5),
                        activations.multiparam_elu_activation((-1*sys.float_info.max/5),
                                                              0.5,0.5))
    assert_almost_equal(activations.multiparam_elu_activation(-0.5,sys.float_info.max_10_exp,0.5),
                        activations.multiparam_elu_activation(-0.5,sys.float_info.max,0.5))
    assert_almost_equal(activations.multiparam_elu_activation(-0.5,0.5,sys.float_info.max_10_exp),
                        activations.multiparam_elu_activation(-0.5,0.5,sys.float_info.max))


def test_weighted_lu():
    assert activations.weighted_lu_activation(-1.0,1.0,1.0) == -1.0
    assert activations.weighted_lu_activation(0.0,1.0,1.0) == 0.0
    assert activations.weighted_lu_activation(1.0,1.0,1.0) == 1.0
    assert activations.weighted_lu_activation(0.0,1.0,0.0) == 0.0
    assert activations.weighted_lu_activation(1.0,1.0,0.0) == 1.0
    assert activations.weighted_lu_activation(-1.0,1.0,-1.0) == 1.0
    assert activations.weighted_lu_activation(0.0,1.0,-1.0) == 0.0
    assert activations.weighted_lu_activation(1.0,1.0,-1.0) == 1.0
    assert activations.weighted_lu_activation(0.0,0.5,1.0) == 0.0
    assert activations.weighted_lu_activation(1.0,0.5,1.0) == 1.0
    assert activations.weighted_lu_activation(0.0,0.5,0.0) == 0.0
    assert activations.weighted_lu_activation(1.0,0.5,0.0) == 1.0
    assert activations.weighted_lu_activation(0.0,0.5,-1.0) == 0.0
    assert activations.weighted_lu_activation(1.0,0.5,-1.0) == 1.0
    assert activations.weighted_lu_activation(0.0,0.0,1.0) == 0.0
    assert activations.weighted_lu_activation(1.0,0.0,1.0) == 1.0
    assert activations.weighted_lu_activation(0.0,0.0,0.0) == 0.0
    assert activations.weighted_lu_activation(1.0,0.0,0.0) == 1.0
    assert activations.weighted_lu_activation(0.0,0.0,-1.0) == 0.0
    assert activations.weighted_lu_activation(1.0,0.0,-1.0) == 1.0
    try:
        ignored = activations.weighted_lu_activation(-1.0,2.0,1.0)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "weighted_lu_activation(-1.0,2.0,1.0) did not raise a ValueError/derived")

def test_multiparam_relu_softplus():
    assert activations.multiparam_relu_softplus_activation(-1.0,1.0,1.0) == 0.0
    assert activations.multiparam_relu_softplus_activation(0.0,1.0,1.0) == 0.0
    assert activations.multiparam_relu_softplus_activation(1.0,1.0,1.0) == 1.0
    assert activations.multiparam_relu_softplus_activation(-1.0,1.0,0.5) == 0.5
    assert activations.multiparam_relu_softplus_activation(0.0,1.0,0.5) == 0.0
    assert activations.multiparam_relu_softplus_activation(1.0,1.0,0.5) == 1.0
    assert activations.multiparam_relu_softplus_activation(-1.0,1.0,0.0) == 1.0
    assert activations.multiparam_relu_softplus_activation(0.0,1.0,0.0) == 0.0
    assert activations.multiparam_relu_softplus_activation(1.0,1.0,0.0) == 1.0
    assert activations.multiparam_relu_softplus_activation(-1.0,0.5,1.0) == -0.5
    assert activations.multiparam_relu_softplus_activation(0.0,0.5,1.0) == 0.0
    assert activations.multiparam_relu_softplus_activation(1.0,0.5,1.0) == 1.0
    assert activations.multiparam_relu_softplus_activation(-1.0,0.0,1.0) == -1.0
    assert activations.multiparam_relu_softplus_activation(0.0,0.0,1.0) == 0.0
    assert activations.multiparam_relu_softplus_activation(1.0,0.0,1.0) == 1.0
    assert_almost_equal(activations.multiparam_relu_softplus_activation(0.0,0.75,0.5),
                        activations.multiparam_relu_softplus_activation(0.0,0.5,0.75))
    assert_almost_equal(activations.multiparam_relu_softplus_activation(0.0,0.5,0.25),
                        activations.multiparam_relu_softplus_activation(0.0,0.25,0.5))
    assert_almost_equal(activations.multiparam_relu_softplus_activation(0.0,0.5,0.0),
                        activations.multiparam_relu_softplus_activation(0.0,0.0,0.5))
    assert_almost_equal(activations.multiparam_relu_softplus_activation(0.0,0.25,0.0),
                        activations.multiparam_relu_softplus_activation(0.0,0.0,0.25))
    assert_almost_equal(activations.multiparam_relu_softplus_activation(0.5,0.5,0.0),
                        activations.multiparam_relu_softplus_activation(0.5,0.0,0.5))
    assert_almost_equal(activations.multiparam_relu_softplus_activation(1.0,0.75,0.5),
                        activations.multiparam_relu_softplus_activation(1.0,0.5,0.75))
    assert_almost_equal(activations.multiparam_relu_softplus_activation(1.0,0.5,0.25),
                        activations.multiparam_relu_softplus_activation(1.0,0.25,0.5))
    assert_almost_equal(activations.multiparam_relu_softplus_activation(1.0,0.5,0.0),
                        activations.multiparam_relu_softplus_activation(1.0,0.0,0.5))
    assert_almost_equal(activations.multiparam_relu_softplus_activation(1.0,0.25,0.0),
                        activations.multiparam_relu_softplus_activation(1.0,0.0,0.25))


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

def test_hat_gauss_rectangular():
    assert activations.hat_gauss_rectangular_activation(-1.0,1.0,1.0) == 0.0
    assert activations.hat_gauss_rectangular_activation(-0.5,1.0,1.0) == 0.5
    assert activations.hat_gauss_rectangular_activation(0.0,1.0,1.0) == 1.0
    assert activations.hat_gauss_rectangular_activation(0.5,1.0,1.0) == 0.5
    assert activations.hat_gauss_rectangular_activation(1.0,1.0,1.0) == 0.0
    assert activations.hat_gauss_rectangular_activation(0.0,1.0,0.75) == 1.0
    assert activations.hat_gauss_rectangular_activation(0.0,1.0,0.25) == 1.0
    assert activations.hat_gauss_rectangular_activation(0.0,1.0,0.0) == 1.0
    assert activations.hat_gauss_rectangular_activation(0.0,0.75,1.0) == 1.0
    assert activations.hat_gauss_rectangular_activation(0.0,0.75,0.5) == 1.0
    assert activations.hat_gauss_rectangular_activation(0.0,0.75,0.0) == 1.0
    assert activations.hat_gauss_rectangular_activation(0.0,0.5,1.0) == 1.0
    assert activations.hat_gauss_rectangular_activation(0.0,0.5,0.75) == 1.0
    assert activations.hat_gauss_rectangular_activation(0.0,0.5,0.5) == 1.0
    assert activations.hat_gauss_rectangular_activation(0.0,0.5,0.25) == 1.0
    assert activations.hat_gauss_rectangular_activation(0.0,0.5,0.0) == 1.0
    assert activations.hat_gauss_rectangular_activation(0.0,0.25,1.0) == 1.0
    assert activations.hat_gauss_rectangular_activation(0.0,0.25,0.5) == 1.0
    assert activations.hat_gauss_rectangular_activation(0.0,0.25,0.0) == 1.0
    assert activations.hat_gauss_rectangular_activation(0.0,0.0,1.0) == 1.0
    assert activations.hat_gauss_rectangular_activation(0.0,0.0,0.75) == 1.0
    assert activations.hat_gauss_rectangular_activation(0.0,0.0,0.5) == 1.0
    assert activations.hat_gauss_rectangular_activation(0.0,0.0,0.25) == 1.0
    assert activations.hat_gauss_rectangular_activation(0.0,0.0,0.0) == 1.0
    assert_almost_equal(activations.hat_gauss_rectangular_activation(-1.0,0.0,0.0),
                        activations.hat_gauss_rectangular_activation(1.0,0.0,0.0))
    assert_almost_equal(activations.hat_gauss_rectangular_activation(-0.5,0.0,0.0),
                        activations.hat_gauss_rectangular_activation(0.5,0.0,0.0))

def test_scaled_expanded_log():
    assert activations.scaled_expanded_log_activation(-1.0,2.0) == -1.0
    assert activations.scaled_expanded_log_activation(1.0,2.0) == 1.0
    assert activations.scaled_expanded_log_activation(-1.0,1.0) == -1.0
    assert activations.scaled_expanded_log_activation(0.0,1.0) <= -6.5
    assert activations.scaled_expanded_log_activation(0.5,1.0) == 0.0
    assert activations.scaled_expanded_log_activation(1.0,1.0) == 1.0
    assert activations.scaled_expanded_log_activation(0.0,0.0) <= -13.0
    assert activations.scaled_expanded_log_activation(1.0,0.0) == 0.0
    assert_almost_equal(activations.scaled_expanded_log_activation(-0.5,1.5),
                        -1*activations.scaled_expanded_log_activation(0.5,1.5))
    assert_almost_equal(activations.scaled_expanded_log_activation(-0.75,1.0),
                        -1*activations.scaled_expanded_log_activation(0.75,1.0))
    assert_almost_equal(activations.scaled_expanded_log_activation(-0.5,0.5),
                        -1*activations.scaled_expanded_log_activation(0.5,0.5))
    assert_almost_equal(activations.scaled_expanded_log_activation(-1.0,0.5),
                        -1*activations.scaled_expanded_log_activation(1.0,0.5))
    assert_almost_equal(activations.scaled_expanded_log_activation(-1.0,1.5),
                        -1*activations.scaled_expanded_log_activation(1.0,1.5))

def test_multiparam_log_inv():
    assert activations.multiparam_log_inv_activation(-1.0,1.0) == -1.0
    assert activations.multiparam_log_inv_activation(-0.5,1.0) == -0.5
    assert activations.multiparam_log_inv_activation(0.0,1.0) == -6.5
    assert activations.multiparam_log_inv_activation(0.5,1.0) == 0.5
    assert activations.multiparam_log_inv_activation(1.0,1.0) == 1.0
    assert activations.multiparam_log_inv_activation(-1.0,0.0) == -1.0
    assert activations.multiparam_log_inv_activation(-0.25,0.0) == 1.0
    assert activations.multiparam_log_inv_activation(0.0,0.0) == -13.0
    assert activations.multiparam_log_inv_activation(0.25,0.0) == -1.0
    assert activations.multiparam_log_inv_activation(0.5,0.0) == 0.0
    assert activations.multiparam_log_inv_activation(1.0,0.0) == 1.0
    assert activations.multiparam_log_inv_activation(-1.0,-0.5) == 0.0
    assert activations.multiparam_log_inv_activation(-0.5,-0.5) == 1.0
    assert activations.multiparam_log_inv_activation(0.0,-0.5) == -6.5
    assert activations.multiparam_log_inv_activation(0.5,-0.5) == -1.0
    assert activations.multiparam_log_inv_activation(1.0,-0.5) == 0.0
    assert activations.multiparam_log_inv_activation(-1.0,-1.0) == 1.0
    assert activations.multiparam_log_inv_activation(-0.5,-1.0) == 2.0
    assert activations.multiparam_log_inv_activation(0.0,-1.0) == 0.0
    assert activations.multiparam_log_inv_activation(0.5,-1.0) == -2.0
    assert activations.multiparam_log_inv_activation(1.0,-1.0) == -1.0
    assert_almost_equal(activations.multiparam_log_inv_activation(-0.5,0.5),
                        -1*activations.multiparam_log_inv_activation(0.5,0.5))
    assert_almost_equal(activations.multiparam_log_inv_activation(-0.75,0.0),
                        -1*activations.multiparam_log_inv_activation(0.75,0.0))
    assert_almost_equal(activations.multiparam_log_inv_activation(-1.0,0.5),
                        -1*activations.multiparam_log_inv_activation(1.0,0.5))

def test_scaled_log1p():
    for n in [0.5,1.0]:
        for m in [0.0,0.5,1.0,1.5,2.0]:
            assert_almost_equal(activations.scaled_log1p_activation(n,m),
                                -1*activations.scaled_log1p_activation(-1*n,m))
    assert activations.scaled_log1p_activation(0.0,2.0) == 0.0
    assert activations.scaled_log1p_activation(0.0,0.0) == 0.0

def test_multiparam_tanh_log1p():
    assert activations.multiparam_tanh_log1p_activation(-1.0,1.0,1.0) == -1.0
    assert activations.multiparam_tanh_log1p_activation(0.0,1.0,1.0) == 0.0
    assert activations.multiparam_tanh_log1p_activation(1.0,1.0,1.0) == 1.0
    assert activations.multiparam_tanh_log1p_activation(0.0,1.0,0.0) == 0.0
    assert activations.multiparam_tanh_log1p_activation(-1.0,1.0,-1.0) == -1.0
    assert activations.multiparam_tanh_log1p_activation(0.0,1.0,-1.0) == 0.0
    assert activations.multiparam_tanh_log1p_activation(1.0,1.0,-1.0) == 1.0
    assert activations.multiparam_tanh_log1p_activation(-1.0,0.5,1.0) == -1.0
    assert activations.multiparam_tanh_log1p_activation(0.0,0.5,1.0) == 0.0
    assert activations.multiparam_tanh_log1p_activation(1.0,0.5,1.0) == 1.0
    assert activations.multiparam_tanh_log1p_activation(0.0,0.5,0.0) == 0.0
    assert activations.multiparam_tanh_log1p_activation(0.0,0.5,-1.0) == 0.0
    assert activations.multiparam_tanh_log1p_activation(-1.0,0.0,1.0) == -1.0
    assert activations.multiparam_tanh_log1p_activation(0.0,0.0,1.0) == 0.0
    assert activations.multiparam_tanh_log1p_activation(1.0,0.0,1.0) == 1.0
    assert activations.multiparam_tanh_log1p_activation(0.0,0.0,0.0) == 0.0
    assert activations.multiparam_tanh_log1p_activation(0.0,0.0,-1.0) == 0.0
    assert_almost_equal(activations.multiparam_tanh_log1p_activation(-0.5,0.0,-1.0),
                        -1*activations.multiparam_tanh_log1p_activation(0.5,0.0,-1.0))
    assert_almost_equal(activations.multiparam_tanh_log1p_activation(-1.0,0.0,-1.0),
                        -1*activations.multiparam_tanh_log1p_activation(1.0,0.0,-1.0))
    assert_almost_equal(activations.multiparam_tanh_log1p_activation(-0.5,0.0,0.0),
                        -1*activations.multiparam_tanh_log1p_activation(0.5,0.0,0.0))
    assert_almost_equal(activations.multiparam_tanh_log1p_activation(-1.0,0.25,-1.0),
                        -1*activations.multiparam_tanh_log1p_activation(1.0,0.25,-1.0))
    assert_almost_equal(activations.multiparam_tanh_log1p_activation(-0.5,0.5,-1.0),
                        -1*activations.multiparam_tanh_log1p_activation(0.5,0.5,-1.0))
    assert_almost_equal(activations.multiparam_tanh_log1p_activation(-1.0,0.0,-0.5),
                        -1*activations.multiparam_tanh_log1p_activation(1.0,0.0,-0.5))
    assert_almost_equal(activations.multiparam_tanh_log1p_activation(-0.5,0.5,0.0),
                        -1*activations.multiparam_tanh_log1p_activation(0.5,0.5,0.0))
    assert_almost_equal(activations.multiparam_tanh_log1p_activation(-1.0,0.5,-1.0),
                        -1*activations.multiparam_tanh_log1p_activation(1.0,0.5,-1.0))
    assert_almost_equal(activations.multiparam_tanh_log1p_activation(-0.5,1.0,0.0),
                        -1*activations.multiparam_tanh_log1p_activation(0.5,1.0,0.0))
    assert_almost_equal(activations.multiparam_tanh_log1p_activation(-1.0,0.5,-0.5),
                        -1*activations.multiparam_tanh_log1p_activation(1.0,0.5,-0.5))
    assert_almost_equal(activations.multiparam_tanh_log1p_activation(-1.0,0.75,-1.0),
                        -1*activations.multiparam_tanh_log1p_activation(1.0,0.75,-1.0))
    assert_almost_equal(activations.multiparam_tanh_log1p_activation(-1.0,0.0,0.0),
                        -1*activations.multiparam_tanh_log1p_activation(1.0,0.0,0.0))
    assert_almost_equal(activations.multiparam_tanh_log1p_activation(-1.0,0.25,0.0),
                        -1*activations.multiparam_tanh_log1p_activation(1.0,0.25,0.0))
    assert_almost_equal(activations.multiparam_tanh_log1p_activation(-1.0,0.5,0.0),
                        -1*activations.multiparam_tanh_log1p_activation(1.0,0.5,0.0))
    assert_almost_equal(activations.multiparam_tanh_log1p_activation(-1.0,0.75,0.0),
                        -1*activations.multiparam_tanh_log1p_activation(1.0,0.75,0.0))
    assert_almost_equal(activations.multiparam_tanh_log1p_activation(-1.0,1.0,0.0),
                        -1*activations.multiparam_tanh_log1p_activation(1.0,1.0,0.0))
    assert_almost_equal(activations.multiparam_tanh_log1p_activation(-1.0,0.0,0.5),
                        -1*activations.multiparam_tanh_log1p_activation(1.0,0.0,0.5))
    assert_almost_equal(activations.multiparam_tanh_log1p_activation(-1.0,0.5,0.5),
                        -1*activations.multiparam_tanh_log1p_activation(1.0,0.5,0.5))
    assert_almost_equal(activations.multiparam_tanh_log1p_activation(-1.0,1.0,0.5),
                        activations.multiparam_tanh_log1p_activation(-1.0,1.0,-0.5))
    assert_almost_equal(activations.multiparam_tanh_log1p_activation(1.0,1.0,0.5),
                        activations.multiparam_tanh_log1p_activation(1.0,1.0,-0.5))

def test_multiparam_pow():
    assert activations.multiparam_pow_activation(-1.0,4.0) == -1.0
    assert activations.multiparam_pow_activation(-0.5,4.0) == -0.0625
    assert activations.multiparam_pow_activation(0.0,4.0) == 0.0
    assert activations.multiparam_pow_activation(0.5,4.0) == 0.0625
    assert activations.multiparam_pow_activation(1.0,4.0) == 1.0
    assert activations.multiparam_pow_activation(-1.0,2.75) == -1.0
    assert activations.multiparam_pow_activation(0.0,2.75) == 0.0
    assert activations.multiparam_pow_activation(1.0,2.75) == 1.0
    assert activations.multiparam_pow_activation(-1.0,1.5) == -1.0
    assert activations.multiparam_pow_activation(-0.25,1.5) == -0.125
    assert activations.multiparam_pow_activation(0.0,1.5) == 0.0
    assert activations.multiparam_pow_activation(0.25,1.5) == 0.125
    assert activations.multiparam_pow_activation(1.0,1.5) == 1.0
    assert activations.multiparam_pow_activation(-1.0,0.25) == -1.0
    assert activations.multiparam_pow_activation(0.0,0.25) == 0.0
    assert activations.multiparam_pow_activation(1.0,0.25) == 1.0
    assert activations.multiparam_pow_activation(-1.0,-1.0) == -1.0
    assert activations.multiparam_pow_activation(0.0,-1.0) == 0.0
    assert activations.multiparam_pow_activation(1.0,-1.0) == 1.0
    assert_almost_equal(activations.multiparam_pow_activation(-0.5,2.75),
                        -1*activations.multiparam_pow_activation(0.5,2.75))
    assert_almost_equal(activations.multiparam_pow_activation(-0.5,1.5),
                        -1*activations.multiparam_pow_activation(0.5,1.5))
    assert_almost_equal(activations.multiparam_pow_activation(-0.75,1.5),
                        -1*activations.multiparam_pow_activation(0.75,1.5))
    assert_almost_equal(activations.multiparam_pow_activation(-0.5,0.25),
                        -1*activations.multiparam_pow_activation(0.5,0.25))
    assert_almost_equal(activations.multiparam_pow_activation(-0.5,-1.0),
                        -1*activations.multiparam_pow_activation(0.5,-1.0))

def test_wave():
    assert activations.wave_activation(0.0,1.0) == 0.0
    assert activations.wave_activation(0.0,0.5) == 0.0
    assert activations.wave_activation(0.0,0.0) == 0.0
    assert activations.wave_activation(0.0,-0.5) == 0.0
    assert activations.wave_activation(-1.0,-1.0) == 1.0
    assert activations.wave_activation(-0.5,-1.0) == -1.0
    assert activations.wave_activation(0.0,-1.0) == 0.0
    assert activations.wave_activation(0.5,-1.0) == 1.0
    assert activations.wave_activation(1.0,-1.0) == -1.0
    assert_almost_equal(activations.wave_activation(-0.5,1.0),-1*activations.wave_activation(0.5,1.0))
    assert_almost_equal(activations.wave_activation(-0.5,0.5),-1*activations.wave_activation(0.5,0.5))
    assert_almost_equal(activations.wave_activation(-0.75,0.0),-1*activations.wave_activation(0.75,0.0))
    assert_almost_equal(activations.wave_activation(-0.5,0.0),-1*activations.wave_activation(0.5,0.0))
    assert_almost_equal(activations.wave_activation(-0.5,-0.5),-1*activations.wave_activation(0.5,-0.5))
    assert_almost_equal(activations.wave_activation(-1.0,1.0),-1*activations.wave_activation(1.0,1.0))
    assert_almost_equal(activations.wave_activation(-1.0,0.5),-1*activations.wave_activation(1.0,0.5))
    assert_almost_equal(activations.wave_activation(-0.25,0.0),-1*activations.wave_activation(0.25,0.0))
    assert_almost_equal(activations.wave_activation(-1.0,0.0),-1*activations.wave_activation(1.0,0.0))
    assert_almost_equal(activations.wave_activation(-1.0,-0.5),-1*activations.wave_activation(1.0,-0.5))

def test_multiparam_tanh_approx():
    assert activations.multiparam_tanh_approx_activation(0.0,1.0,1.0) == 0.0
    assert activations.multiparam_tanh_approx_activation(0.0,1.0,0.5) == 0.0
    assert activations.multiparam_tanh_approx_activation(0.0,1.0,0.0) == 0.0
    assert activations.multiparam_tanh_approx_activation(0.0,0.25,1.0) == 0.0
    assert activations.multiparam_tanh_approx_activation(0.0,0.25,0.75) == 0.0
    assert activations.multiparam_tanh_approx_activation(0.0,0.25,0.5) == 0.0
    assert activations.multiparam_tanh_approx_activation(0.0,0.25,0.25) == 0.0
    assert activations.multiparam_tanh_approx_activation(0.0,0.25,0.0) == 0.0
    assert activations.multiparam_tanh_approx_activation(0.0,0.0,1.0) == 0.0
    assert activations.multiparam_tanh_approx_activation(0.0,0.0,0.75) == 0.0
    assert activations.multiparam_tanh_approx_activation(0.0,0.0,0.5) == 0.0
    assert activations.multiparam_tanh_approx_activation(0.0,0.0,0.25) == 0.0
    assert activations.multiparam_tanh_approx_activation(-1.0,0.0,0.0) == -2.5
    assert activations.multiparam_tanh_approx_activation(-0.5,0.0,0.0) == -1.25
    assert activations.multiparam_tanh_approx_activation(0.0,0.0,0.0) == 0.0
    assert activations.multiparam_tanh_approx_activation(0.5,0.0,0.0) == 1.25
    assert activations.multiparam_tanh_approx_activation(1.0,0.0,0.0) == 2.5
    assert activations.multiparam_tanh_approx_activation(0.0,-0.5,1.0) == 0.0
    assert activations.multiparam_tanh_approx_activation(0.0,-0.5,0.75) == 0.0
    assert activations.multiparam_tanh_approx_activation(0.0,-0.5,0.5) == 0.0
    assert activations.multiparam_tanh_approx_activation(0.0,-0.5,0.25) == 0.0
    assert activations.multiparam_tanh_approx_activation(0.0,-0.5,0.0) == 0.0
    assert activations.multiparam_tanh_approx_activation(0.0,-1.0,1.0) == 0.0
    assert activations.multiparam_tanh_approx_activation(0.0,-1.0,0.5) == 0.0
    assert activations.multiparam_tanh_approx_activation(0.0,-1.0,0.0) == 0.0
    assert_almost_equal(activations.multiparam_tanh_approx_activation(-1.0,1.0,1.0),
                        -1*activations.multiparam_tanh_approx_activation(1.0,1.0,1.0))
    assert_almost_equal(activations.multiparam_tanh_approx_activation(-0.5,0.25,1.0),
                        -1*activations.multiparam_tanh_approx_activation(0.5,0.25,1.0))
    assert_almost_equal(activations.multiparam_tanh_approx_activation(-0.5,0.0,1.0),
                        -1*activations.multiparam_tanh_approx_activation(0.5,0.0,1.0))
    assert_almost_equal(activations.multiparam_tanh_approx_activation(-1.0,1.0,0.5),
                        -1*activations.multiparam_tanh_approx_activation(1.0,1.0,0.5))
    assert_almost_equal(activations.multiparam_tanh_approx_activation(-0.5,0.25,0.5),
                        -1*activations.multiparam_tanh_approx_activation(0.5,0.25,0.5))
    assert_almost_equal(activations.multiparam_tanh_approx_activation(-1.0,0.25,1.0),
                        -1*activations.multiparam_tanh_approx_activation(1.0,0.25,1.0))
    assert_almost_equal(activations.multiparam_tanh_approx_activation(-0.5,-0.5,1.0),
                        -1*activations.multiparam_tanh_approx_activation(0.5,-0.5,1.0))
    assert_almost_equal(activations.multiparam_tanh_approx_activation(-1.0,0.0,1.0),
                        -1*activations.multiparam_tanh_approx_activation(1.0,0.0,1.0))
    assert_almost_equal(activations.multiparam_tanh_approx_activation(-0.5,0.0,0.5),
                        -1*activations.multiparam_tanh_approx_activation(0.5,0.0,0.5))
    assert_almost_equal(activations.multiparam_tanh_approx_activation(-1.0,0.25,0.75),
                        -1*activations.multiparam_tanh_approx_activation(1.0,0.25,0.75))
    assert_almost_equal(activations.multiparam_tanh_approx_activation(-1.0,-0.5,1.0),
                        -1*activations.multiparam_tanh_approx_activation(1.0,-0.5,1.0))
    assert_almost_equal(activations.multiparam_tanh_approx_activation(-1.0,0.0,0.75),
                        -1*activations.multiparam_tanh_approx_activation(1.0,0.0,0.75))
    assert_almost_equal(activations.multiparam_tanh_approx_activation(-1.0,-1.0,1.0),
                        -1*activations.multiparam_tanh_approx_activation(1.0,-1.0,1.0))
    assert_almost_equal(activations.multiparam_tanh_approx_activation(-1.0,1.0,0.0),
                        -1*activations.multiparam_tanh_approx_activation(1.0,1.0,0.0))
    assert_almost_equal(activations.multiparam_tanh_approx_activation(-0.5,0.25,0.0),
                        -1*activations.multiparam_tanh_approx_activation(0.5,0.25,0.0))
    assert_almost_equal(activations.multiparam_tanh_approx_activation(-1.0,0.25,0.5),
                        -1*activations.multiparam_tanh_approx_activation(1.0,0.25,0.5))
    assert_almost_equal(activations.multiparam_tanh_approx_activation(-1.0,-0.5,0.75),
                        -1*activations.multiparam_tanh_approx_activation(1.0,-0.5,0.75))
    assert_almost_equal(activations.multiparam_tanh_approx_activation(-0.5,-0.5,0.5),
                        -1*activations.multiparam_tanh_approx_activation(0.5,-0.5,0.5))
    assert_almost_equal(activations.multiparam_tanh_approx_activation(-1.0,0.0,0.5),
                        -1*activations.multiparam_tanh_approx_activation(1.0,0.0,0.5))
    assert_almost_equal(activations.multiparam_tanh_approx_activation(-1.0,0.25,0.25),
                        -1*activations.multiparam_tanh_approx_activation(1.0,0.25,0.25))
    assert_almost_equal(activations.multiparam_tanh_approx_activation(-1.0,-0.5,0.5),
                        -1*activations.multiparam_tanh_approx_activation(1.0,-0.5,0.5))
    assert_almost_equal(activations.multiparam_tanh_approx_activation(-1.0,0.0,0.25),
                        -1*activations.multiparam_tanh_approx_activation(1.0,0.0,0.25))
    assert_almost_equal(activations.multiparam_tanh_approx_activation(-1.0,-1.0,0.5),
                        -1*activations.multiparam_tanh_approx_activation(1.0,-1.0,0.5))
    assert_almost_equal(activations.multiparam_tanh_approx_activation(-1.0,0.25,0.0),
                        -1*activations.multiparam_tanh_approx_activation(1.0,0.25,0.0))
    assert_almost_equal(activations.multiparam_tanh_approx_activation(-1.0,-0.5,0.25),
                        -1*activations.multiparam_tanh_approx_activation(1.0,-0.5,0.25))
    assert_almost_equal(activations.multiparam_tanh_approx_activation(-0.5,-0.5,0.0),
                        -1*activations.multiparam_tanh_approx_activation(0.5,-0.5,0.0))
    assert_almost_equal(activations.multiparam_tanh_approx_activation(-1.0,-0.5,0.0),
                        -1*activations.multiparam_tanh_approx_activation(1.0,-0.5,0.0))
    assert_almost_equal(activations.multiparam_tanh_approx_activation(-1.0,-1.0,0.0),
                        -1*activations.multiparam_tanh_approx_activation(1.0,-1.0,0.0))

def test_multiparam_sigmoid_approx():
    assert activations.multiparam_sigmoid_approx_activation(0.0,1.0) == 0.5
    assert activations.multiparam_sigmoid_approx_activation(0.0,0.5) == 0.5
    assert activations.multiparam_sigmoid_approx_activation(0.0,0.0) == 0.5
    assert activations.multiparam_sigmoid_approx_activation(0.0,-0.5) == 0.5
    assert activations.multiparam_sigmoid_approx_activation(0.0,-1.0) == 0.5

def test_multiparam_gauss():
    for a in [0.0, 0.5, 1.0, 2.0, 3.0, 4.0]:
        for b in [0.0, 0.5, 1.0, 1.5, 2.0]:
            assert activations.multiparam_gauss_activation(0.0,a,b) == 1.0
            for z in [0.25,0.5,0.75,1.0]:
                assert_almost_equal(activations.multiparam_gauss_activation(z,a,b),
                                    activations.multiparam_gauss_activation(-z,a,b))

def test_function_set():
    m = multiparameter.MultiParameterSet('activation')
    s = activations.ActivationFunctionSet(m)
    assert s.get('sigmoid') is not None
    assert s.get('tanh') is not None
    assert s.get('sin') is not None
    assert s.get('gauss') is not None
    assert s.get('relu') is not None
    assert s.get('identity') is not None
    assert s.get('clamped') is not None
    assert s.get('inv') is not None
    assert s.get('log') is not None
    assert s.get('expanded_log') is not None
    assert s.get('skewed_log1p') is not None
    assert s.get('log1p') is not None
    assert s.get('exp') is not None
    assert s.get('abs') is not None
    assert s.get('hat') is not None
    assert s.get('square') is not None
    assert s.get('cube') is not None
    assert s.get('square_wave') is not None
    assert s.get('triangle_wave') is not None
    assert s.get('rectangular') is not None
    assert m.get_MPF('multiparam_relu', 'activation') is not None
    assert m.get_MPF('multiparam_relu_softplus', 'activation') is not None
    assert m.get_MPF('multiparam_elu', 'activation') is not None
    assert m.get_MPF('weighted_lu', 'activation') is not None
    assert m.get_MPF('clamped_tanh_step', 'activation') is not None
    assert m.get_MPF('multiparam_sigmoid', 'activation') is not None
    assert m.get_MPF('hat_gauss_rectangular', 'activation') is not None
    assert m.get_MPF('scaled_expanded_log', 'activation') is not None
    assert m.get_MPF('multiparam_log_inv', 'activation') is not None
    assert m.get_MPF('scaled_log1p', 'activation') is not None
    assert m.get_MPF('multiparam_tanh_log1p', 'activation') is not None
    assert m.get_MPF('multiparam_pow', 'activation') is not None
    assert m.get_MPF('wave', 'activation') is not None
    assert m.get_MPF('multiparam_tanh_approx', 'activation') is not None
    assert m.get_MPF('multiparam_sigmoid_approx', 'activation') is not None
    assert m.get_MPF('multiparam_gauss', 'activation') is not None

    assert s.is_valid('sigmoid')
    assert s.is_valid('tanh')
    assert s.is_valid('sin')
    assert s.is_valid('gauss')
    assert s.is_valid('relu')
    assert s.is_valid('identity')
    assert s.is_valid('clamped')
    assert s.is_valid('inv')
    assert s.is_valid('log')
    assert s.is_valid('expanded_log')
    assert s.is_valid('skewed_log1p')
    assert s.is_valid('log1p')
    assert s.is_valid('exp')
    assert s.is_valid('abs')
    assert s.is_valid('hat')
    assert s.is_valid('square')
    assert s.is_valid('cube')
    assert s.is_valid('square_wave')
    assert s.is_valid('triangle_wave')
    assert s.is_valid('rectangular')
    assert s.is_valid('multiparam_relu')
    assert s.is_valid('multiparam_relu_softplus')
    assert s.is_valid('multiparam_elu')
    assert s.is_valid('weighted_lu')
    assert s.is_valid('clamped_tanh_step')
    assert s.is_valid('multiparam_sigmoid')
    assert s.is_valid('hat_gauss_rectangular')
    assert s.is_valid('scaled_expanded_log')
    assert s.is_valid('multiparam_log_inv')
    assert s.is_valid('scaled_log1p')
    assert s.is_valid('multiparam_tanh_log1p')
    assert s.is_valid('multiparam_pow')
    assert s.is_valid('wave')
    assert s.is_valid('multiparam_tanh_approx')
    assert s.is_valid('multiparam_sigmoid_approx')
    assert s.is_valid('multiparam_gauss')

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

    assert config.genome_config.get_activation_MPF('multiparam_relu') is not None
    assert config.genome_config.get_activation_MPF('multiparam_relu_softplus') is not None
    assert config.genome_config.get_activation_MPF('multiparam_elu') is not None
    assert config.genome_config.get_activation_MPF('weighted_lu') is not None
    assert config.genome_config.get_activation_MPF('clamped_tanh_step') is not None
    assert config.genome_config.get_activation_MPF('multiparam_sigmoid') is not None

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

    assert config.genome_config.get_activation_Evolved_MPF('multiparam_relu') is not None
    assert config.genome_config.get_activation_Evolved_MPF('multiparam_relu_softplus') is not None
    assert config.genome_config.get_activation_Evolved_MPF('multiparam_elu') is not None
    assert config.genome_config.get_activation_Evolved_MPF('weighted_lu') is not None
    assert config.genome_config.get_activation_Evolved_MPF('clamped_tanh_step') is not None
    assert config.genome_config.get_activation_Evolved_MPF('multiparam_sigmoid') is not None

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

    assert config.genome_config.get_activation_Evolved_MPF('multiparam_relu(0.5)') is not None
    assert config.genome_config.get_activation_Evolved_MPF('multiparam_relu_softplus(0.5,0.5)') is not None
    assert config.genome_config.get_activation_Evolved_MPF('multiparam_elu(0.5,0.5)') is not None
    assert config.genome_config.get_activation_Evolved_MPF('weighted_lu(0.5,0.5)') is not None
    assert config.genome_config.get_activation_Evolved_MPF('clamped_tanh_step(0.5)') is not None
    assert config.genome_config.get_activation_Evolved_MPF('multiparam_sigmoid(0.5)') is not None

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

    test_result = config.genome_config.get_activation_Evolved_MPF('multiparam_relu(0.5)')
    assert str(test_result) == str(config.genome_config.get_activation_Evolved_MPF(str(test_result)))
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
    test_function_set()
    test_get_MPF()
    test_get_Evolved_MPF_simple()
    test_get_Evolved_MPF_complex()
    test_bad_add1()
    test_bad_add2()
